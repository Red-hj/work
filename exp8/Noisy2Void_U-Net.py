import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image

device = torch.device("cuda")
log_dir = r"E:\explore\Noisy2Void\logs_train"
result_dir = r"E:\explore\Noisy2Void\result"
model_path = r"E:\explore\Noisy2Void\model"
div2k_path = r"E:\explore\data\DIV2K"

batch_size = 16
epochs = 200
learning_rate = 1e-4
weight_decay = 1e-8
sigma = 25
repeat = 10
writer = SummaryWriter(log_dir=log_dir)


def pad_to_16(x):
    h, w = x.shape[-2:]
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    return nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')


train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(pad_to_16)
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Lambda(pad_to_16)
])


class DIV2KDataset(Dataset):
    def __init__(self, root, train=True, transform=None, repeat=10):
        self.transform = transform
        self.img_paths = []
        self.repeat = repeat if train else 1
        folder = os.path.join(root, "DIV2K_train_HR" if train else "DIV2K_valid_HR")
        for name in os.listdir(folder):
            if name.endswith(('png', 'jpg')):
                self.img_paths.append(os.path.join(folder, name))

    def __len__(self):
        return len(self.img_paths) * self.repeat

    def __getitem__(self, idx):
        actual_idx = idx % len(self.img_paths)
        img = Image.open(self.img_paths[actual_idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        # 生成高斯噪声图
        noise = torch.randn_like(img) * (sigma / 255.0)
        noisy = img + noise
        noisy = torch.clamp(noisy, 0.0, 1.0)
        return noisy, img


def apply_n2v_mask(noisy, mask_prob=0.05, neighbor_range=2):
    b, c, h, w = noisy.shape
    mask = torch.rand((b, c, h, w), device=noisy.device) < mask_prob

    shifts = [(dy, dx) for dy in range(-neighbor_range, neighbor_range + 1)
              for dx in range(-neighbor_range, neighbor_range + 1)
              if dy != 0 or dx != 0]

    rand_indices = torch.randint(0, len(shifts), (b, c, h, w), device=noisy.device)

    noisy_shifted = torch.empty_like(noisy)

    for i, (dy, dx) in enumerate(shifts):
        shifted = torch.roll(noisy, shifts=(dy, dx), dims=(2, 3))
        noisy_shifted = torch.where(rand_indices == i, shifted, noisy_shifted)

    input_masked = torch.where(mask, noisy_shifted, noisy)
    return input_masked, mask


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self.conv_block(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        self.out = nn.Conv2d(64, out_channels, 1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)


# 标准训练函数
def train(model, criterion, optimizer, loader, epoch):
    model.train()
    total_loss = 0
    total_psnr = 0
    total_imgs = 0
    start = time.time()

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        input_masked, mask = apply_n2v_mask(noisy, mask_prob=0.05)

        mask_count = mask.sum()

        if mask_count == 0:
            continue

        optimizer.zero_grad()
        denoised_pred = model(input_masked)

        loss = criterion(denoised_pred[mask], noisy[mask]) / mask_count

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        with torch.no_grad():
            denoised_clamped = torch.clamp(denoised_pred, 0.0, 1.0)
            mse = torch.mean((denoised_clamped - clean) ** 2, dim=[1, 2, 3])
            batch_psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
            total_psnr += batch_psnr.sum().item()

        total_loss += loss.item() * noisy.size(0)
        total_imgs += noisy.size(0)

    avg_loss = total_loss / total_imgs if total_imgs > 0 else 0
    avg_psnr = total_psnr / total_imgs if total_imgs > 0 else 0
    cost = time.time() - start

    writer.add_scalar("Train/Loss", avg_loss, epoch)
    writer.add_scalar("Train/PSNR", avg_psnr, epoch)
    print(f"Epoch {epoch + 1:2d} | Loss {avg_loss:.6f} | PSNR {avg_psnr:.2f} dB | {cost:.1f}s")
    return avg_psnr


# 测试函数
def test(model, loader):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    total_imgs = 0

    with torch.no_grad():
        for noisy, clean in loader:
            clean = clean.to(device)
            noisy = noisy.to(device)

            denoised = model(noisy)
            denoised = torch.clamp(denoised, 0.0, 1.0)

            clean_np = clean.cpu().float().numpy()
            denoised_np = denoised.cpu().float().numpy()

            for c, d in zip(clean_np, denoised_np):
                c = c.squeeze()
                d = d.squeeze()
                total_psnr += psnr(c, d, data_range=1.0)
                total_ssim += ssim(c, d, data_range=1.0, channel_axis=None)
            total_imgs += clean.size(0)

    avg_psnr = total_psnr / total_imgs
    avg_ssim = total_ssim / total_imgs
    print(f"测试PSNR: {avg_psnr:.2f} dB   SSIM: {avg_ssim:.4f}\n")
    return avg_psnr, avg_ssim


if __name__ == "__main__":
    train_set = DIV2KDataset(div2k_path, train=True, transform=train_transform, repeat=repeat)
    test_set = DIV2KDataset(div2k_path, train=False, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    model = UNet(in_channels=1, out_channels=1).to(device)

    criterion = nn.MSELoss(reduction='sum')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_psnr = 0.0
    best_ssim = 0.0
    best_ssim_epoch = 0
    best_psnr_epoch = 0

    print("开始训练\n")
    for epoch in range(epochs):
        train_psnr = train(model, criterion, optimizer, train_loader, epoch)
        test_psnr, test_ssim = test(model, test_loader)
        scheduler.step()

        if test_psnr > best_psnr:
            best_psnr = test_psnr
            best_psnr_epoch = epoch
            torch.save(model.state_dict(), os.path.join(model_path, "n2v_best.pth"))
            print(f"模型已保存 | Best PSNR = {best_psnr:.2f} dB\n")

        if test_ssim > best_ssim:
            best_ssim = test_ssim
            best_ssim_epoch = epoch

    print(f"Best PSNR: {best_psnr:.2f} dB in {best_psnr_epoch + 1:2d} epoch")
    print(f"Best SSIM: {best_ssim:.4f} in {best_ssim_epoch + 1:2d} epoch")
    writer.close()
