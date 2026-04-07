import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image

device = torch.device("cuda")

log_dir = r"E:\explore\Noisy2Void\logs_train"
result_dir = r"E:\explore\Noisy2Void\result"
zaoseng_dir = r"E:\explore\Noisy2Void\zaoseng"
model_path = r"E:\explore\Noisy2Void\model"
set14_path = r"E:\explore\data\set14"

sigma = 25
batch_size = 1



def pad_to_16(x):
    h, w = x.shape[-2:]
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    return nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')


test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Lambda(pad_to_16)
])


class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.img_paths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(('png', 'jpg'))]
        self.img_names = [f for f in os.listdir(root) if f.endswith(('png', 'jpg'))]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, idx


def add_noise(image, sigma=25):
    noise = torch.randn_like(image) * (sigma / 255.0)
    noisy = image + noise
    noisy = torch.clamp(noisy, 0.0, 1.0)
    return noisy


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


def test(model, loader):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    total_time = 0
    total_imgs = 0

    with torch.no_grad():
        for data, idx in loader:
            clean = data.to(device)
            noisy = add_noise(clean, sigma)

            t0 = time.time()
            denoised = model(noisy)
            t1 = time.time()
            infer_time = t1 - t0

            denoised = torch.clamp(denoised, 0.0, 1.0)

            c = clean.squeeze().cpu().numpy()
            d = denoised.squeeze().cpu().numpy()

            p = psnr(c, d, data_range=1)
            s = ssim(c, d, data_range=1, channel_axis=None)

            name = loader.dataset.img_names[idx.item()]
            base = os.path.splitext(name)[0]

            noisy_np = (noisy.squeeze().cpu().numpy() * 255).astype(np.uint8)
            denoised_np = (denoised.squeeze().cpu().numpy() * 255).astype(np.uint8)

            Image.fromarray(noisy_np).save(os.path.join(zaoseng_dir, f'{sigma}_N2V_noisy_{base}.png'))
            Image.fromarray(denoised_np).save(os.path.join(result_dir, f'{sigma}_N2V_denoised_{base}.png'))

            print(f'{name:15s} | time:{infer_time:6.4f}s | PSNR:{p:6.2f} | SSIM:{s:6.4f}')

            total_psnr += p
            total_ssim += s
            total_time += infer_time
            total_imgs += 1

    print(
        f'\nAvg time:{total_time / total_imgs:.4f}s | Avg PSNR:{total_psnr / total_imgs:.2f} | AvgSSIM:{total_ssim / total_imgs:.4f}')


if __name__ == "__main__":
    test_set = TestDataset(set14_path, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, "n2v_best.pth"), map_location=device))

    print("开始测试\n")
    test(model, test_loader)
