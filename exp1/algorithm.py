import os
import cv2
import numpy as np
import bm3d
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

DATASET_PATH = r"E:\shuju"
NOISY_SAVE_PATH = r"E:\zaoseng"
RESULT_SAVE_PATH = r"E:\bm"
GAUSSIAN_SIGMA = 25  # 高斯噪声σ=25
SALT_PEPPER_AMOUNT = 0.02  # 椒盐噪声


#噪声添加
def add_mixed_noise(image, sigma=25,salt_pepper_amount=0.02):
    #添加高斯噪声
    gaussian_noise = np.random.normal(0, sigma / 255.0, image.shape)
    noisy_img = image + gaussian_noise
    noisy_img = np.clip(noisy_img, 0, 1)

    #添加椒盐噪声
    noisy_img_sp = np.copy(noisy_img)
    #盐噪声
    num_salt = np.ceil(salt_pepper_amount * image.size * 0.5).astype(int)
    coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    noisy_img_sp[coords[0], coords[1]] = 1.0
    #胡椒噪声
    num_pepper = np.ceil(salt_pepper_amount * image.size * 0.5).astype(int)
    coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy_img_sp[coords[0], coords[1]] = 0.0

    return noisy_img_sp

for img_name in os.listdir(DATASET_PATH):
    img_path = os.path.join(DATASET_PATH, img_name)

    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图像: {img_name}")
        continue

    # 归一化
    img = img.astype(np.float32) / 255.0
    img_name_base = os.path.splitext(img_name)[0]

    # 添加混合噪声
    noisy_img = add_mixed_noise(img, GAUSSIAN_SIGMA, SALT_PEPPER_AMOUNT)

    # 保存含噪图像
    noisy_save_name = f"{img_name_base}_mixed_noisy.png"
    noisy_save_path = os.path.join(NOISY_SAVE_PATH, noisy_save_name)
    cv2.imwrite(noisy_save_path, (noisy_img * 255).astype(np.uint8))

    # BM3D去噪
    denoised_img = bm3d.bm3d(noisy_img, sigma_psd=GAUSSIAN_SIGMA / 255.0)
    denoised_img = np.clip(denoised_img, 0, 1)

    # 保存去噪后的图像
    denoised_save_name = f"{img_name_base}_mixed_denoised.png"
    denoised_save_path = os.path.join(RESULT_SAVE_PATH, denoised_save_name)
    cv2.imwrite(denoised_save_path, (denoised_img * 255).astype(np.uint8))

    # 计算PSNR和SSIM
    psnr_value = psnr(img, denoised_img, data_range=1.0)
    ssim_value = ssim(img, denoised_img, data_range=1.0)

    print(f"处理完成: {img_name} (mixed)")
    print(f"   PSNR: {psnr_value:.2f} dB | SSIM: {ssim_value:.4f}")

print("\n")
print("-全部处理完成-")
