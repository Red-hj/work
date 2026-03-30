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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_dir = r"E:\explore\FFDNet\logs_train"
result_dir = r"E:\explore\FFDNet\result"
model_path = r"E:\explore\FFDNet\model"
div2k_path = r"E:\explore\data\DIV2K"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

batch_size = 32
epochs = 100
learning_rate = 1e-3
weight_decay = 0
sigma = 25
repeat = 10

writer = SummaryWriter(log_dir=log_dir)


def pad_to_even(x):
    h, w = x.shape[-2:]
    pad_h = (2 - h % 2) % 2
    pad_w = (2 - w % 2) % 2
    return nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')


train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomCrop(128),
    transforms.ToTensor(),
    transforms.Lambda(pad_to_even)
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Lambda(pad_to_even)
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
        return img, 0


class FFDNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64):
        super(FFDNet, self).__init__()

        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv1 = nn.Conv2d(in_channels * 4 + 1, num_features, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        layers = []
        for _ in range(13):
            layers.append(nn.Conv2d(num_features, num_features, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*layers)

        self.conv_last = nn.Conv2d(num_features, out_channels * 4, 3, padding=1, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, sigma):
        x = self.pixel_unshuffle(x)

        sigma_map = sigma.expand(x.size(0), 1, x.size(2), x.size(3))
        x = torch.cat([x, sigma_map], dim=1)

        x = self.relu(self.conv1(x))
        x = self.body(x)
        x = self.conv_last(x)

        x = self.pixel_shuffle(x)
        return x


def train(model, criterion, optimizer, loader, epoch):
    model.train()
    total_loss = 0
    total_psnr = 0
    total_imgs = 0
    start = time.time()

    for data, _ in loader:
        clean = data.to(device)

        noise = torch.randn_like(clean) * (sigma / 255.0)
        noisy = clean + noise

        sigma_tensor = torch.full((clean.size(0), 1, 1, 1), sigma / 255.0, device=device)

        optimizer.zero_grad()
        noise_pred = model(noisy, sigma_tensor)

        loss = criterion(noise_pred, noise)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            denoised = noisy - noise_pred
            denoised_clamped = torch.clamp(denoised, 0.0, 1.0)

            mse = torch.mean((denoised_clamped - clean) ** 2, dim=[1, 2, 3])
            batch_psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
            total_psnr += batch_psnr.sum().item()

        total_loss += loss.item() * clean.size(0)
        total_imgs += clean.size(0)

    avg_loss = total_loss / total_imgs
    avg_psnr = total_psnr / total_imgs
    cost = time.time() - start

    writer.add_scalar("Train/Loss", avg_loss, epoch)
    writer.add_scalar("Train/PSNR", avg_psnr, epoch)
    print(f"Epoch {epoch + 1:2d} | Loss {avg_loss:.6f} | PSNR {avg_psnr:.2f} dB | {cost:.1f}s")
    return avg_psnr


def test(model, loader):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    total_imgs = 0

    with torch.no_grad():
        for data, _ in loader:
            clean = data.to(device)

            noise = torch.randn_like(clean) * (sigma / 255.0)
            noisy = clean + noise

            sigma_tensor = torch.full((clean.size(0), 1, 1, 1), sigma / 255.0, device=device)
            noise_pred = model(noisy, sigma_tensor)

            denoised = noisy - noise_pred
            denoised = torch.clamp(denoised, 0.0, 1.0)

            clean_np = clean.cpu().numpy()
            denoised_np = denoised.cpu().numpy()

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

    model = FFDNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

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
            torch.save(model.state_dict(), os.path.join(model_path, "ffdnet_best.pth"))
            print(f"模型已保存 | Best PSNR = {best_psnr:.2f} dB\n")

        if test_ssim > best_ssim:
            best_ssim = test_ssim
            best_ssim_epoch = epoch

    print(f"Best PSNR: {best_psnr:.2f} dB in {best_psnr_epoch + 1:2d} epoch")
    print(f"Best SSIM: {best_ssim:.4f} in {best_ssim_epoch + 1:2d} epoch")
    writer.close()
