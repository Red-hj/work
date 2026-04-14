import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image

device = torch.device("cuda")
log_dir = r"E:\explore\Neighbor2Neighbor\logs_train"
result_dir = r"E:\explore\Neighbor2Neighbor\re sult"
model_path = r"E:\explore\Neighbor2Neighbor\model"
div2k_path = r"E:\explore\data\DIV2K"

batch_size = 16
epochs = 200
learning_rate = 1e-4
weight_decay = 1e-5
sigma = 25
repeat = 10
gamma = 0.5

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
        noise = torch.randn_like(img) * (sigma / 255.0)
        noisy = img + noise
        noisy = torch.clamp(noisy, 0.0, 1.0)
        return noisy, img

def neighbor_sampling_fast(noisy):
    #随机偏移采样
    r1, c1 = torch.randint(0, 2, (1,)).item(), torch.randint(0, 2, (1,)).item()
    y1 = noisy[:, :, r1::2, c1::2]
    y2 = noisy[:, :, (1 - r1)::2, (1 - c1)::2]
    if torch.rand(1) > 0.5:
        return y1, y2, (r1, c1), (1 - r1, 1 - c1)
    return y2, y1, (1 - r1, 1 - c1), (r1, c1)

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
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
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
        noise_pred = self.out(d1)
        return x - noise_pred

def train(model, criterion, optimizer, loader, epoch):
    model.train()
    total_loss, total_psnr, total_imgs = 0, 0, 0
    start = time.time()

    for noisy, clean in loader:
        noisy = noisy.to(device)
        y1, y2, pos1, _ = neighbor_sampling_fast(noisy)

        optimizer.zero_grad()
        out1 = model(y1)
        out2 = model(y2)

        loss_rec = criterion(out1, y2) + criterion(out2, y1)
        loss_reg = criterion(out1 - out2, y1 - y2)
        if epoch < 20:
            gamma = 0.5
        elif epoch < 60:
            gamma = 0.2
        else:
            gamma = 0.1
        loss = loss_rec + gamma * loss_reg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        with torch.no_grad():
            r, c = pos1
            clean_sub = clean[:, :, r::2, c::2].to(device)
            mse = torch.mean((torch.clamp(out1, 0, 1) - clean_sub) ** 2, dim=[1, 2, 3])
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

def test(model, loader):
    model.eval()
    total_psnr, total_ssim, total_imgs = 0, 0, 0
    with torch.no_grad():
        for noisy, clean in loader:
            clean, noisy = clean.to(device), noisy.to(device)
            denoised = torch.clamp(model(noisy), 0, 1)
            c_np, d_np = clean.cpu().numpy().squeeze(), denoised.cpu().numpy().squeeze()
            total_psnr += psnr(c_np, d_np, data_range=1.0)
            total_ssim += ssim(c_np, d_np, data_range=1.0, channel_axis=None)
            total_imgs += 1

    avg_psnr, avg_ssim = total_psnr / total_imgs, total_ssim / total_imgs
    print(f"测试PSNR: {avg_psnr:.2f} dB   SSIM: {avg_ssim:.4f}\n")
    return avg_psnr, avg_ssim

if __name__ == "__main__":
    train_set = DIV2KDataset(div2k_path, train=True, transform=train_transform, repeat=repeat)
    test_set = DIV2KDataset(div2k_path, train=False, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    model = UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)
    writer = SummaryWriter(log_dir=log_dir)

    best_psnr, best_ssim = 0.0, 0.0
    best_psnr_epoch, best_ssim_epoch = 0, 0
    print("开始训练\n")

    for epoch in range(epochs):
        train_psnr = train(model, criterion, optimizer, train_loader, epoch)
        test_psnr, test_ssim = test(model, test_loader)
        scheduler.step(test_psnr)

        if test_psnr > best_psnr:
            best_psnr, best_psnr_epoch = test_psnr, epoch
            torch.save(model.state_dict(), os.path.join(model_path, "n2n_best.pth"))
            print(f"模型已保存 | Best PSNR = {best_psnr:.2f} dB\n")
        if test_ssim > best_ssim:
            best_ssim, best_ssim_epoch = test_ssim, epoch

    print(f"Best PSNR: {best_psnr:.2f} dB in {best_psnr_epoch + 1:2d} epoch")
    print(f"Best SSIM: {best_ssim:.4f} in {best_ssim_epoch + 1:2d} epoch")
    writer.close()
