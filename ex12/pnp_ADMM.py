import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_data_path = r"E:\explore\data\DIV2K\DIV2K_valid_HR"
model_path = r"E:\explore\U-Net\model\unet_best.pth"
save_path = r"E:\explore\PnP_ADMM\result"

sigma = 25
max_iter = 12
rho_dn = 150.0
rho_mri = 0.08
eta = 0.6

def fft2(x):
    return torch.fft.fftshift(torch.fft.fft2(x))

def ifft2(x):
    return torch.fft.ifft2(torch.fft.ifftshift(x))

def pad_img(x):
    h, w = x.shape[-2:]
    ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
    return nn.functional.pad(x, (0, pw, 0, ph), mode='reflect')


test_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

class TestSet(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.files = sorted([f for f in os.listdir(path) if f.endswith(('png', 'jpg'))])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.path, self.files[idx])).convert('L')
        return self.transform(img) if self.transform else img

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = self.block(in_channels, 64); self.enc2 = self.block(64, 128)
        self.enc3 = self.block(128, 256); self.enc4 = self.block(256, 512)
        self.pool = nn.MaxPool2d(2); self.bottleneck = self.block(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2); self.dec4 = self.block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2); self.dec3 = self.block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2); self.dec2 = self.block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2); self.dec1 = self.block(128, 64)
        self.out = nn.Conv2d(64, out_channels, 1)
    def block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        identity = x
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return identity - self.out(d1)

def denoise(noisy, net, rho, max_iter, sigma, eta):
    sigma_n = sigma / 255.0
    x, v = noisy.clone(), noisy.clone()
    u = torch.zeros_like(x)
    alpha = rho * (sigma_n ** 2)
    for _ in range(max_iter):
        x = (noisy + alpha * (v - u)) / (1.0 + alpha)
        with torch.no_grad():
            v = net(x + u)
        v = torch.clamp(v, 0.0, 1.0)
        u = u + eta * (x - v)
    return v

def mri_recon(clean, net, mask, rho, max_iter, eta):
    k_clean = fft2(clean)
    k_under = k_clean * mask
    under = ifft2(k_under).real
    x, v = under.clone(), under.clone()
    u = torch.zeros_like(x)
    for _ in range(max_iter):
        fft_vu = fft2(v - u)
        fxt = (k_under + rho * fft_vu) / (1.0 + rho) * mask + fft_vu * (1 - mask)
        x = ifft2(fxt).real
        with torch.no_grad():
            v = net(x + u)
        v = torch.clamp(v, 0.0, 1.0)
        u = u + eta * (x - v)
    return v, under


if __name__ == "__main__":
    if not os.path.exists(save_path): os.makedirs(save_path)
    net = UNet().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    test_set = TestSet(test_data_path, test_transform)
    loader = DataLoader(test_set, batch_size=1, shuffle=False)

    mask = torch.zeros(1, 1, 256, 256, device=device)
    mask[:, :, :, 96:160] = 1

    sum_dn_psnr, sum_dn_ssim, sum_mri_psnr, sum_mri_ssim = 0, 0, 0, 0
    cnt = 0

    start = time.time()
    with torch.no_grad():
        for clean in loader:
            clean = clean.to(device)
            cnt += 1
            if cnt > 100: break

            sigma_n = sigma / 255.0
            noisy = torch.clamp(clean + torch.randn_like(clean) * sigma_n, 0.0, 1.0)

            denoised = denoise(noisy, net, rho_dn, max_iter, sigma, eta)
            mri_rec, mri_under = mri_recon(clean, net, mask, rho_mri, max_iter, eta)

            c_np, n_np, d_np = clean.squeeze().cpu().numpy(), noisy.squeeze().cpu().numpy(), denoised.squeeze().cpu().numpy()
            mu_np, mr_np = mri_under.squeeze().cpu().numpy(), mri_rec.squeeze().cpu().numpy()

            if cnt <= 5:
                cv2.imwrite(os.path.join(save_path, f"noisy_{cnt:02d}.png"), (n_np * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(save_path, f"denoised_{cnt:02d}.png"), (d_np * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(save_path, f"mri_under_{cnt:02d}.png"), (mu_np * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(save_path, f"mri_recon_{cnt:02d}.png"), (mr_np * 255).astype(np.uint8))

            dn_psnr, dn_ssim = psnr(c_np, d_np, data_range=1.0), ssim(c_np, d_np, data_range=1.0)
            mri_psnr, mri_ssim = psnr(c_np, mr_np, data_range=1.0), ssim(c_np, mr_np, data_range=1.0)

            sum_dn_psnr += dn_psnr; sum_dn_ssim += dn_ssim
            sum_mri_psnr += mri_psnr; sum_mri_ssim += mri_ssim

            print(f"[{cnt:02d}] 去噪 PSNR: {dn_psnr:.2f} | MRI PSNR: {mri_psnr:.2f}")

    print(f"去噪 (σ=25)")
    print(f"  PSNR: {sum_dn_psnr / cnt:.2f} dB")
    print(f"  SSIM: {sum_dn_ssim / cnt:.4f}")
    print(f"MRI 重建 (2倍欠采样)")
    print(f"  PSNR: {sum_mri_psnr / cnt:.2f} dB")
    print(f"  SSIM: {sum_mri_ssim / cnt:.4f}")
    print(f"总耗时: {time.time() - start:.1f}s")
