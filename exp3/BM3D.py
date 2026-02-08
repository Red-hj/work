import os
import cv2
import numpy as np
import bm3d
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

date = r"E:\explore\set14"
save = r"E:\explore\bm\zaoseng"
result = r"E:\explore\bm\result"

'''
gaussian_sigma = 35  # 高斯噪声σ=35
salt_pepper_amount = 0.03  # 椒盐噪声
'''

missing_ratio = 0.3 #随机缺失像素比例

#噪声添加
'''
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
'''


def generate_missing_pixels(image,missing_ratio=0.3):
    img = np.copy(image)
    h, w = img.shape
    total_pixels = h * w
    missing_pixels = int(total_pixels * missing_ratio)

    row_idx = np.random.randint(0,h,missing_pixels)
    col_idx = np.random.randint(0,w,missing_pixels)

    img[row_idx, col_idx] = 0
    mask = np.ones_like(img)
    mask[row_idx, col_idx] = 0

    return img,mask

def bm3d_inpainting(missing_img,mask):
    temp_restore = bm3d.bm3d(missing_img,sigma_psd=0.01)
    inpainted_img = missing_img * mask + temp_restore * (1 - mask)
    inpainted_img = np.clip(inpainted_img,0,1)

    return inpainted_img


total_start = time.time()
single_time_list = []

for name in os.listdir(date):

    single_start = time.time()
    img_path = os.path.join(date,name)

    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

    #归一化
    img = img.astype(np.float32) / 255.0
    img_name_base = os.path.splitext(name)[0]

    #添加混合噪声
    '''
    noisy_img = add_mixed_noise(img,gaussian_sigma,salt_pepper_amount)

    # 保存含噪图像
    noisy_name = f"{img_name_base}_noisy.png"
    noisy_path = os.path.join(save, noisy_name)
    cv2.imwrite(noisy_path,(noisy_img * 255).astype(np.uint8))

    # BM3D去噪
    denoised = bm3d.bm3d(noisy_img,sigma_psd=gaussian_sigma / 255.0)
    denoised = np.clip(denoised,0,1)

    # 保存去噪后的图像
    denoised_save_name = f"{img_name_base}_denoised.png"
    denoised_save_path = os.path.join(result, denoised_save_name)
    cv2.imwrite(denoised_save_path,(denoised * 255).astype(np.uint8))
    '''
    #生成缺失像素图像
    missing_img, mask = generate_missing_pixels(img,missing_ratio)

    # 保存含缺失像素的图像
    missing_name = f"{img_name_base}_noisy.png"
    missing_path = os.path.join(save,missing_name)
    cv2.imwrite(missing_path, (missing_img * 255).astype(np.uint8))

    # BM3D图像补全
    inpainted = bm3d_inpainting(missing_img,mask)
    inpainted = np.clip(inpainted,0,1)

    # 保存补全后的图像
    inpainted_save_name = f"{img_name_base}_denoised.png"
    inpainted_save_path = os.path.join(result,inpainted_save_name)
    cv2.imwrite(inpainted_save_path, (inpainted * 255).astype(np.uint8))

    # 计算PSNR和SSIM
    psnr_value = psnr(img,inpainted,data_range=1.0)
    ssim_value = ssim(img,inpainted,data_range=1.0)

    single_end = time.time()
    single_cost = round(single_end - single_start,4)
    single_time_list.append(single_cost)

    print(f"{name}")
    print(f"---PSNR: {psnr_value:.2f} dB --- SSIM: {ssim_value:.4f}")
    print(f"处理时长: {single_cost:.4f} 秒\n")

total_end = time.time()
total_cost = round(total_end - total_start,4)
avg_cost = round(np.mean(single_time_list),4) if single_time_list else 0.0

print("\n")
print(f"总时长: {total_cost:.4f} 秒")
print(f"平均处理时长: {avg_cost:.4f} 秒")
