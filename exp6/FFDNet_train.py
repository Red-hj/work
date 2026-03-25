import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Dataset

device = torch.device("cuda")

log_dir = r"E:\explore\FFDNet\logs_train"
result_dir = r"E:\explore\FFDNet\result"
model_path = r"E:\explore\FFDNet\model"

batch_size = 32
epochs = 100
learning_rate = 1e-3
weight_decay = 1e-4
sigma_min = 5
sigma_max = 50
salt_pepper_amount = 0.03


writer = SummaryWriter(log_dir=log_dir)

def pad_to_even(x):
    h, w = x.shape[-2:]
    pad_h = (2 - h % 2) % 2
    pad_w = (2 - w % 2) % 2
    return nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(pad_to_even)
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

def add_noisy(image, sigma=25):
    if isinstance(image, torch.Tensor):
        img_np = image.cpu().numpy().squeeze(1)
    else:
        img_np = image.copy()

    gaussian_noise = np.random.normal(0, sigma / 255.0, img_np.shape)
    noisy_img = img_np + gaussian_noise
    noisy_img = np.clip(noisy_img, -1, 1)

    if isinstance(image, torch.Tensor):
        ans = torch.from_numpy(noisy_img).unsqueeze(1).float().to(device)
        return ans
    return noisy_img

class FFDNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64):
        super(FFDNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels * 4 + 1, num_features, kernel_size=3, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        layers = []

        for _ in range(18):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))

        self.body = nn.Sequential(*layers)
        self.conv_last = nn.Conv2d(num_features, out_channels * 4, kernel_size=3, padding=1, bias=False)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_and_sigma):
        x, sigma = x_and_sigma
        batch_size, in_channels, h, w = x.size()

        x = x.view(batch_size, in_channels, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(batch_size, in_channels * 4, h // 2, w // 2)
        sigma_map = sigma.repeat(1, 1, h // 2, w // 2)
        x = torch.cat((x, sigma_map), 1)
        x = self.relu1(self.conv1(x))
        x = self.body(x)
        x = self.conv_last(x)

        out_channels = x.size(1) // 4
        x = x.view(batch_size, out_channels, 2, 2, h // 2, w // 2)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(batch_size, out_channels, h, w)

        return x

def train(model, criterion, optimizer, train_loader, epoch):
    model.train()

    epoch_start = time.time()
    train_psnr = 0.0
    train_ssim = 0.0
    batch_count = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        clean_img = data.to(device)
        sigma_val = np.random.uniform(sigma_min, sigma_max)
        noisy_img = add_noisy(clean_img, sigma_val)
        sigma_tensor = torch.full((clean_img.size(0), 1, 1, 1), sigma_val / 255.0).to(device)
        denoised_img = model((noisy_img, sigma_tensor))
        loss = criterion(denoised_img, clean_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        clean_np = (clean_img.cpu().detach().numpy() + 1) / 2
        denoised_np = (denoised_img.cpu().detach().numpy() + 1) / 2

        for c, d in zip(clean_np, denoised_np):
            c = c.squeeze()
            d = d.squeeze()
            train_psnr += psnr(c, d, data_range=1.0)
            train_ssim += ssim(c, d, data_range=1.0, channel_axis=None)

        batch_count += 1

    avg_train_psnr = train_psnr / (batch_count * batch_size)
    avg_train_ssim = train_ssim / (batch_count * batch_size)

    epoch_time = round(time.time() - epoch_start, 4)

    writer.add_scalar('Train/Loss', loss.item(), epoch)
    writer.add_scalar('Train/PSNR', avg_train_psnr, epoch)
    writer.add_scalar('Train/SSIM', avg_train_ssim, epoch)
    writer.add_scalar('Train/Epoch_Time', epoch_time, epoch)

    print(f"第 {epoch + 1}轮训练\n")
    print(f"Loss: {loss.item():.4f}   PSNR: {avg_train_psnr:.2f} dB   SSIM: {avg_train_ssim:.4f}\n")
    print(f"训练耗时: {epoch_time:.4f} s\n")

    return avg_train_psnr, avg_train_ssim

def test(model, criterion, test_loader):
    model.eval()

    test_psnr = 0.0
    test_ssim = 0.0
    img_count = 0

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            clean_img = data.to(device)
            sigma_test = 35
            noisy_img = add_noisy(clean_img, sigma_test)
            noisy_img_save = noisy_img
            sigma_tensor = torch.full((clean_img.size(0), 1, 1, 1), sigma_test / 255.0).to(device)
            denoised_img = model((noisy_img, sigma_tensor))
            clean_np = (clean_img[0].cpu().numpy().squeeze() + 1) / 2 * 255
            noisy_np = (noisy_img_save[0].cpu().numpy().squeeze() + 1) / 2 * 255
            denoised_np = (denoised_img[0].cpu().numpy().squeeze() + 1) / 2 * 255
            cv2.imwrite(os.path.join(result_dir, f"test_clean_{batch_idx}.png"), clean_np.astype(np.uint8))
            cv2.imwrite(os.path.join(result_dir, f"test_noisy_{batch_idx}.png"), noisy_np.astype(np.uint8))
            cv2.imwrite(os.path.join(result_dir, f"test_denoised_{batch_idx}.png"), denoised_np.astype(np.uint8))
            clean_np = (clean_img.cpu().numpy() + 1) / 2
            denoised_np = (denoised_img.cpu().numpy() + 1) / 2

            for c, d in zip(clean_np, denoised_np):
                c = c.squeeze()
                d = d.squeeze()
                test_psnr += psnr(c, d, data_range=1.0)
                test_ssim += ssim(c, d, data_range=1.0, channel_axis=None)

            img_count += 1

    avg_test_psnr = test_psnr / (img_count * batch_size)
    avg_test_ssim = test_ssim / (img_count * batch_size)
    print(f"----测试 PSNR: {avg_test_psnr:.2f} dB  SSIM: {avg_test_ssim:.4f}")
    return avg_test_psnr, avg_test_ssim

if __name__ == "__main__":
    model = FFDNet(in_channels=1, out_channels=1).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print("开始训练...")
    total_train_time = time.time()

    for epoch in range(epochs):
        avg_train_psnr, avg_train_ssim = train(model, criterion, optimizer, train_loader, epoch)

    total_train_time = round(time.time() - total_train_time, 4)
    writer.add_scalar('Train/Total_Time', total_train_time)
    print(f"总时长: {total_train_time:.4f} s")
    torch.save(model.state_dict(), os.path.join(model_path, "ffdnet.pth"))
    print("\n开始测试...")
    avg_test_psnr, avg_test_ssim = test(model, criterion, test_loader)
    writer.add_hparams({
        'lr': learning_rate,
        'batch_size': batch_size,
        'sigma_min': sigma_min,
        'sigma_max': sigma_max
    }, {
        'final_train_psnr': avg_train_psnr,
        'final_train_ssim': avg_train_ssim,
        'final_test_psnr': avg_test_psnr,
        'final_test_ssim': avg_test_ssim
    })
    writer.close()
