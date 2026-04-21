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

log_dir = r"E:\explore\DIP\logs_train"
bsd68_path = r"E:\explore\data\BSD68\original"
result_dir = r"E:\explore\DIP\result2"

sigmas = [15,25,35,50]
num_iterations = 5000
learning_rate = 1e-3
patience = 2500


def pad_to_16(x):
    h, w = x.shape[-2:]
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    return nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')


def add_noise(image, sigma=25):
    noise = torch.randn_like(image).to(device) * (sigma / 255.0)
    noisy = image + noise
    return torch.clamp(noisy, 0.0, 1.0)

class DIPUNet(nn.Module):
    def __init__(self, input_depth=32, output_depth=1):
        super(DIPUNet, self).__init__()

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Encoder
        self.conv1 = nn.Sequential(nn.Conv2d(input_depth, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True))

        # Decoder
        self.deconv3 = nn.Sequential(nn.Conv2d(512+256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
        self.deconv2 = nn.Sequential(nn.Conv2d(256+128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
        self.deconv1 = nn.Sequential(nn.Conv2d(128+64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True))

        self.out = nn.Conv2d(64, output_depth, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))

        up3 = self.up(conv4)
        concat3 = torch.cat([up3, conv3], dim=1)
        deconv3 = self.deconv3(concat3)

        up2 = self.up(deconv3)
        concat2 = torch.cat([up2, conv2], dim=1)
        deconv2 = self.deconv2(concat2)

        up1 = self.up(deconv2)
        concat1 = torch.cat([up1, conv1], dim=1)
        deconv1 = self.deconv1(concat1)

        return self.sigmoid(self.out(deconv1))


def get_noise(input_depth, spatial_size):
    return torch.randn(1, input_depth, spatial_size[0], spatial_size[1])

if __name__ == "__main__":
    writer = SummaryWriter(log_dir=log_dir)
    img_list = [f for f in os.listdir(bsd68_path) if f.endswith(('.png', '.jpg'))]

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(pad_to_16)
    ])

    for sigma in sigmas:
        total_psnr = 0.0
        total_ssim = 0.0
        print(f"---Sigma = {sigma} ---")

        for idx, img_name in enumerate(img_list):
            if idx >= 13: break

            img_path = os.path.join(bsd68_path, img_name)
            clean = transform(Image.open(img_path)).unsqueeze(0).to(device)
            noisy = add_noise(clean, sigma)

            model = DIPUNet(input_depth=32, output_depth=1).to(device)
            net_input = get_noise(32, (clean.shape[2], clean.shape[3])).to(device) * 0.1
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            best_psnr_val = 0
            best_denoised_img = None
            no_optim_count = 0

            start_time = time.time()
            for i in range(num_iterations):
                model.train()
                output = model(net_input)
                loss = criterion(output, noisy)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 50 == 0:
                    model.eval()
                    with torch.no_grad():
                        current_out = model(net_input)
                        out_np = current_out.squeeze().cpu().numpy()
                        clean_np = clean.squeeze().cpu().numpy()
                        current_psnr = psnr(clean_np, out_np, data_range=1)

                    if current_psnr > best_psnr_val:
                        best_psnr_val = current_psnr
                        best_denoised_img = current_out.detach().clone()
                        no_optim_count = 0
                    else:
                        no_optim_count += 50

                    if no_optim_count >= patience:
                        print(f"Early Stopping at Iteration {i + 1}")
                        break

                if (i + 1) % 500 == 0:
                    print(f"Image {idx + 1} [{img_name}] | Iter {i + 1}/{num_iterations} | Loss: {loss.item():.6f}")

            denoised = torch.clamp(best_denoised_img, 0.0, 1.0)
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
            torch.cuda.empty_cache()

        num_imgs = min(len(img_list), 13)
        avg_psnr = total_psnr / num_imgs
        avg_ssim = total_ssim / num_imgs
        print(f"Average PSNR: {avg_psnr:.2f} dB | Average SSIM: {avg_ssim:.4f}\n")

    writer.close()
