import os
import cv2
import numpy as np
import time
import pywt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

date = r"E:\explore\set14"
save = r"E:\explore\ADMM\zaosheng"
result = r"E:\explore\ADMM\result"

ratio = 0.3 #随机缺失像素比例
lambda_ = 0.05 #正则化系数
rho = 0.1 #ADMM惩罚参数（降低rho增强迭代效果）
iter = 3000 #最大迭代次数
tol = 1e-5 #收敛阈值

#生成随机像素缺失的图像
def generate_missing_pixels(image,ratio=0.3):
    img = np.copy(image)
    h,w = img.shape
    total = h*w
    pixels = int(total * ratio)
    row_idx = np.random.randint(0,h,pixels)
    col_idx = np.random.randint(0,w,pixels)
    img[row_idx, col_idx] = 0
    mask = np.ones_like(img)
    mask[row_idx, col_idx] = 0
    return img,mask

#软阈值函数
def soft_threshold(x,threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold,0)

#ADMM算法实现图像补全
def admm(img,mask):
    x = np.ones_like(img) * 0.5 #初始化改为0.5（避免初始等于缺失图像）
    z = np.copy(x)
    u = np.zeros_like(x)
    prev_x = np.copy(x)
    convergence = []
    original_shape = img.shape

    for i in range(iter):
        #1 x-update (最小化关于x的增广拉格朗日函数)
        numerator = mask*img + rho*(z - u)
        denominator = mask + rho + 1e-8 #防止除0
        x = numerator / denominator
        x = np.clip(x,0,1)

        #2 z-update (小波域软阈值，L1正则项)
        coeffs = pywt.dwt2(x + u,'db4')
        cA,(cH,cV,cD) = coeffs
        #高频系数软阈值处理（调整阈值计算）
        threshold = lambda_ / (rho * 10) #降低阈值增强细节
        cH_update = soft_threshold(cH,threshold)
        cV_update = soft_threshold(cV,threshold)
        cD_update = soft_threshold(cD,threshold)
        #逆小波变换还原z
        z = pywt.idwt2((cA,(cH_update,cV_update,cD_update)),'db4')
        z = z[:original_shape[0], :original_shape[1]]
        z = np.clip(z,0,1)

        #3 u-update (对偶变量更新)
        u = u + 0.01 * (x - z) #加入步长控制更新幅度

        #收敛判断
        residual = np.linalg.norm(x - prev_x)/(np.linalg.norm(prev_x) + 1e-8)
        convergence.append(residual)
        if residual < tol:
            print(f"迭代{i+1}次后收敛")
            break
        prev_x = np.copy(x)

    return x,convergence

total_start = time.time()
single_time_list = []

#批量处理图像
for name in os.listdir(date):
    single_start = time.time()
    img_path = os.path.join(date,name)

    #读取灰度图像 归一化
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)/255.0
    base = os.path.splitext(name)[0]

    # 生成缺失像素图像
    missing_img, mask = generate_missing_pixels(img,ratio)

    # 保存含缺失像素的图像
    missing_name = f"{base}_noisy.png"
    missing_path = os.path.join(save,missing_name)
    cv2.imwrite(missing_path,(missing_img*255).astype(np.uint8))

    # ADMM图像补全
    inpainted_img,convergence = admm(missing_img,mask)
    inpainted_img = np.clip(inpainted_img,0,1)

    # 保存补全后的图像
    inpainted_save_name = f"{base}_denoised.png"
    inpainted_save_path = os.path.join(result,inpainted_save_name)
    cv2.imwrite(inpainted_save_path,(inpainted_img*255).astype(np.uint8))

    # 计算PSNR和SSIM
    psnr_value = psnr(img,inpainted_img,data_range=1.0)
    ssim_value = ssim(img,inpainted_img,data_range=1.0)

    # 计时与日志输出
    single_end = time.time()
    single_cost = round(single_end-single_start,4)
    single_time_list.append(single_cost)

    print(f"{name}")
    print(f"---PSNR: {psnr_value:.2f} dB --- SSIM: {ssim_value:.4f}")
    print(f"迭代次数: {len(convergence)}   最终收敛残差: {convergence[-1]:.6f}")
    print(f"处理时长: {single_cost:.4f} 秒\n")

# 总计输出
total_end = time.time()
total_cost = round(total_end-total_start,4)
avg_cost = round(np.mean(single_time_list),4) if single_time_list else 0.0

print(f"总时长: {total_cost:.4f} 秒")
print(f"平均处理时长: {avg_cost:.4f} 秒")
