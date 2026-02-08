import os
import cv2
import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pywt

date = r"E:\explore\set14"
save = r"E:\explore\ISTA\zaosheng"
result = r"E:\explore\ISTA\result"

'''
gaussian_sigma = 25  # 高斯噪声σ=25
salt_pepper_amount = 0.02  # 椒盐噪声
'''

ratio = 0.3 #随机缺失像素比例(30%)
lambda_ = 0.05 #正则化系数
iter = 3000 #最大迭代次数
slen = 0.002 #步长
tol = 1e-5 #收敛值

"""
# 噪声添加
def noise_add(image,sigma=25,salt=0.02):
    #添加高斯噪声
    gaussian_noise = np.random.normal(0,sigma/255.0,image.shape)
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
"""

#生成随机像素缺失的图像
def generate_missing_pixels(image,ratio=0.3):
    img = np.copy(image)
    h, w = img.shape
    total = h * w
    pixels = int(total * ratio)

    row_idx = np.random.randint(0,h,pixels)
    col_idx = np.random.randint(0,w,pixels)

    img[row_idx,col_idx] = 0
    mask = np.ones_like(img)
    mask[row_idx,col_idx] = 0

    return img,mask


def soft_threshold(x,threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


"""
def ista(noisy_img):
    x = np.copy(noisy_img)
    prev_x = np.copy(x)
    convergence = []
    # 保存原始图像尺寸
    original_shape = noisy_img.shape

    for i in range(iter):
        #计算图像空间的梯度
        grad = x-noisy_img
        #小波变换
        coeffs = pywt.dwt2(x,'db4')
        cA,(cH,cV,cD) = coeffs
        #对高频系数做软阈值
        cH_update = soft_threshold(cH,lambda_ *slen)
        cV_update = soft_threshold(cV,lambda_ *slen)
        cD_update = soft_threshold(cD,lambda_ *slen)
        #逆变换
        denoised_x = pywt.idwt2((cA,(cH_update,cV_update,cD_update)),'db4')
        #裁剪回原始尺寸
        denoised_x = denoised_x[:original_shape[0], :original_shape[1]]
        denoised_x = np.clip(denoised_x,0,1)
        x = denoised_x - slen * grad
        # 计算残差判断收敛性
        residual = np.linalg.norm(x-prev_x)/np.linalg.norm(prev_x)
        convergence.append(residual)
        if residual < tol:
            print(f"迭代{i+1}次后收敛")
            break
        prev_x = np.copy(x)
    return x, convergence
"""


def ista(img,mask):
    x = np.copy(img)
    prev_x = np.copy(x)
    convergence = []
    original_shape = img.shape

    for i in range(iter):
        #计算图像空间的梯度
        grad = (x - img) * mask
        #小波变换
        coeffs = pywt.dwt2(x,'db4')
        cA, (cH, cV, cD) = coeffs
        #对高频系数做软阈值
        cH_update = soft_threshold(cH,lambda_ * slen)
        cV_update = soft_threshold(cV,lambda_ * slen)
        cD_update = soft_threshold(cD,lambda_ * slen)
        #逆变换
        inpainted_x = pywt.idwt2((cA, (cH_update,cV_update,cD_update)),'db4')
        #裁剪回原始尺寸
        inpainted_x = inpainted_x[:original_shape[0], :original_shape[1]]
        inpainted_x = np.clip(inpainted_x,0,1)
        x = inpainted_x - slen * grad
        #计算残差判断收敛性
        residual = np.linalg.norm(x - prev_x)/np.linalg.norm(prev_x)
        convergence.append(residual)
        if residual < tol:
            print(f"迭代{i + 1}次后收敛")
            break
        prev_x = np.copy(x)
    return x,convergence


total_start = time.time()
single_time_list = []

for name in os.listdir(date):
    single_start = time.time()
    img_path = os.path.join(date,name)

    #读取图像
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    #归一化
    img = img.astype(np.float32)/255.0
    base = os.path.splitext(name)[0]

    #生成缺失像素图像
    missing_img, mask = generate_missing_pixels(img,ratio)

    #保存含缺失像素的图像
    missing_name = f"{base}_noisy.png"
    missing_path = os.path.join(save,missing_name)
    cv2.imwrite(missing_path, (missing_img * 255).astype(np.uint8))
    #ISTA
    inpainted, convergence = ista(missing_img,mask)
    inpainted = np.clip(inpainted, 0, 1)

    #保存补全后的图像
    save_name = f"{base}_denoised.png"
    save_path = os.path.join(result,save_name)
    cv2.imwrite(save_path, (inpainted * 255).astype(np.uint8))

    #计算PSNR和SSIM
    psnr_value = psnr(img,inpainted,data_range=1.0)
    ssim_value = ssim(img,inpainted,data_range=1.0)

    single_end = time.time()
    single_cost = round(single_end - single_start,4)
    single_time_list.append(single_cost)

    print(f"{name}")
    print(f"---PSNR: {psnr_value:.2f} dB --- SSIM: {ssim_value:.4f}")
    print(f"迭代次数: {len(convergence)}   最终收敛残差: {convergence[-1]:.6f}")
    print(f"处理时长: {single_cost:.4f} 秒\n")

total_end = time.time()
total_cost = round(total_end - total_start,4)
avg_cost = round(np.mean(single_time_list),4) if single_time_list else 0.0

print(f"总时长: {total_cost:.4f} 秒")
print(f"平均处理时长: {avg_cost:.4f} 秒")
