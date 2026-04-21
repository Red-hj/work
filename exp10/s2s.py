import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.utils as vutils

device = torch.device("cuda")


log_dir = r"E:\explore\Self2Self\logs_train"
result_dir = r"E:\explore\Self2Self\result2"
bsd68_path = r"E:\explore\data\BSD68\original"

sigmas = [15, 25, 35, 50]
num_iterations = 7000

learning_rate = 1e-4
patience = 1200
ensemble_times = 30


def pad_to_16(x):
    h, w = x.shape[-2:]
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    return nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')


def add_noise(image, sigma=25):
    noise = torch.randn_like(image).to(device) * (sigma / 255.0)
    noisy = image + noise
    return torch.clamp(noisy, 0.0, 1.0)


class Self2SelfUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dropout_p=0.3):
        super().__init__()
        self.dropout_p = dropout_p
        self.enc1 = self.double_conv(in_channels, 32)
        self.enc2 = self.double_conv(32, 64)
        self.enc3 = self.double_conv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self.double_conv(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self.double_conv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self.double_conv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self.double_conv(64, 32)
        self.out = nn.Sequential(
            nn.Conv2d(32, out_channels, 1),
            nn.Sigmoid()
        )

    def double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = torch.cat([self.up3(b), e3], dim=1)
        d3 = self.dec3(d3)
        d2 = torch.cat([self.up2(d3), e2], dim=1)
        d2 = self.dec2(d2)
        d1 = torch.cat([self.up1(d2), e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)


def ensemble_denoise(model, x, times=30):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    outs = []
    with torch.no_grad():
        for _ in range(times):
            outs.append(model(x))
    return torch.mean(torch.stack(outs), dim=0)


if __name__ == "__main__":
    writer = SummaryWriter(log_dir=log_dir)
    img_list = [f for f in os.listdir(bsd68_path) if f.endswith(('.png', '.jpg'))]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(pad_to_16)
    ])

    for sigma in sigmas:
        total_psnr, total_ssim = 0.0, 0.0
        print(f"---Sigma = {sigma} ---")

        for idx, img_name in enumerate(img_list):
            if idx >= 13: break

            img_path = os.path.join(bsd68_path, img_name)
            clean = transform(Image.open(img_path)).unsqueeze(0).to(device)
            noisy = add_noise(clean, sigma)

            model = Self2SelfUNet(in_channels=1).to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            best_psnr_val, wait = 0.0, 0
            start_time = time.time()

            for i in range(num_iterations):
                model.train()
                mask = torch.bernoulli(torch.full_like(noisy, 0.7)).to(device)
                model_input = noisy * mask
                output = model(model_input)

                loss = torch.sum(((output - noisy) ** 2) * (1 - mask)) / (torch.sum(1 - mask) + 1e-8)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    model.eval()
                    with torch.no_grad():

                        current_out = ensemble_denoise(model, noisy, times=10)
                        out_np = torch.clamp(current_out, 0, 1).squeeze().cpu().numpy()
                        clean_np = clean.squeeze().cpu().numpy()
                        current_psnr_val = psnr(clean_np, out_np, data_range=1)

                    if current_psnr_val > best_psnr_val:
                        best_psnr_val = current_psnr_val
                    #     wait = 0
                    # else:
                    #     wait += 50
                    #
                    # if wait >= patience:
                    #     print(f"Early Stopping at Iteration {i + 1}")
                    #     break

                if (i + 1) % 500 == 0:

                    print(f"Image {idx + 1} [{img_name}] | Iter {i + 1}/{num_iterations} | Loss: {loss.item():.6f}")

            denoised = torch.clamp(ensemble_denoise(model, noisy, times=ensemble_times), 0.0, 1.0)
            c_np = clean.squeeze().cpu().numpy()
            d_np = denoised.squeeze().cpu().numpy()

            curr_psnr = psnr(c_np, d_np, data_range=1)
            curr_ssim = ssim(c_np, d_np, data_range=1)

            total_psnr += curr_psnr
            total_ssim += curr_ssim

            cost = time.time() - start_time

            print(f"{img_name} | PSNR: {curr_psnr:.2f} | SSIM: {curr_ssim:.4f} | Time: {cost:.1f}s\n")

            img_idx = f"{idx + 1:02d}"
            sigma_str = str(sigma)

            vutils.save_image(noisy, os.path.join(result_dir, f"{img_idx}_{sigma_str}_noisy.jpg"))
            vutils.save_image(denoised, os.path.join(result_dir, f"{img_idx}_{sigma_str}_denoised.jpg"))

            writer.add_scalar(f"Sigma{sigma}/PSNR", curr_psnr, idx)
            comparison = torch.cat([clean[0], noisy[0], denoised[0]], dim=2)
            writer.add_image(f"Sigma{sigma}/Result_{idx}", comparison, 0)

            del model, optimizer

        num_imgs = min(len(img_list), 13)
        avg_psnr = total_psnr / num_imgs
        avg_ssim = total_ssim / num_imgs
        print(f"Average PSNR: {avg_psnr:.2f} dB | Average SSIM: {avg_ssim:.4f}\n")

    writer.close()
