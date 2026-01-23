import os
import cv2
import numpy as np
import bm3d
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

date = r"E:\shuju"
save = r"E:\zaoseng"
result = r"E:\bm"
gaussian_sigma = 25  # 高斯噪声σ=25
salt_pepper_amount = 0.02  # 椒盐噪声


#噪声添加
def add_mixed_noise(image,sigma=25,salt=0.02):
    #添加高斯噪声
    gaussian_noise = np.random.normal(0,sigma / 255.0, image.shape)
    noisy_img = image + gaussian_noise
    noisy_img = np.clip(noisy_img,0,1)

    #添加椒盐噪声
    ans = np.copy(noisy_img)
    #盐噪声
    num1 = np.ceil(salt * image.size * 0.5).astype(int)
    t = [np.random.randint(0,i,num1) for i in image.shape]
    ans[t[0], t[1]] = 1.0
    #胡椒噪声
    num2 = np.ceil(salt * image.size * 0.5).astype(int)
    t = [np.random.randint(0,i,num2) for i in image.shape]
    ans[t[0],t[1]] = 0.0

    return ans

for name in os.listdir(date):
    img_path = os.path.join(date,name)

    # 读取图像
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

    # 归一化
    img = img.astype(np.float32) / 255.0
    img_name_base = os.path.splitext(name)[0]

    # 添加混合噪声
    noisy_img = add_mixed_noise(img,gaussian_sigma,salt_pepper_amount)

    # 保存含噪图像
    noisy_name = f"{img_name_base}_noisy.png"
    noisy_path = os.path.join(save, noisy_name)
    cv2.imwrite(noisy_path,(noisy_img * 255).astype(np.uint8))

    # BM3D去噪
    denoised = bm3d.bm3d(noisy_img,sigma_psd=gaussian_sigma / 255.0)
    denoised = np.clip(denoised,0,1)

    # 保存去噪后的图像
    denoised_save_name = f"{img_name_base}_mixed_denoised.png"
    denoised_save_path = os.path.join(result, denoised_save_name)
    cv2.imwrite(denoised_save_path,(denoised * 255).astype(np.uint8))

    # 计算PSNR和SSIM
    psnr_value = psnr(img,denoised,data_range=1.0)
    ssim_value = ssim(img,denoised,data_range=1.0)

    print(f"{name}")
    print(f"---PSNR: {psnr_value:.2f} dB --- SSIM: {ssim_value:.4f}")

print("\n")
print("-全部处理完成-")
