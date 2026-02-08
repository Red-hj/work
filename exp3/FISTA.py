import os
import cv2
import numpy as np
import time
import pywt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

date = r"E:\explore\set14"
save = r"E:\explore\FISTA\zaosheng"
result = r"E:\explore\FISTA\result"

ratio = 0.3 #随机缺失像素比例
lambda_ = 0.05 #正则化系数
iter = 2000 #最大迭代次数
slen = 0.001 #步长
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

def soft_threshold(x,threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold,0)

#TV正则化的梯度计算
def tv_gradient(x):
    h,w = x.shape
    # 水平方向梯度
    grad_h = np.zeros_like(x)
    grad_h[:-1, :] = x[1:, :] - x[:-1, :]
    # 垂直方向梯度
    grad_v = np.zeros_like(x)
    grad_v[:, :-1] = x[:, 1:] - x[:, :-1]
    # 反向梯度
    grad_tv = np.zeros_like(x)
    grad_tv[:-1, :] -= grad_h[:-1, :]
    grad_tv[1:, :] += grad_h[:-1, :]
    grad_tv[:, :-1] -= grad_v[:, :-1]
    grad_tv[:, 1:] += grad_v[:, :-1]
    return grad_tv

def fista(img,mask):
    x = np.copy(img)
    y = np.copy(x)
    t = 1.0
    prev_x = np.copy(x)
    convergence = []
    original_shape = img.shape

    for i in range(iter):
        #计算残差
        res = (y - img) * mask
        #L1正则化 小波变换+软阈值
        coeffs = pywt.dwt2(y,'db4')
        cA,(cH,cV,cD) = coeffs
        #高频系数软阈值处理
        cH_update = soft_threshold(cH,lambda_ * slen)
        cV_update = soft_threshold(cV,lambda_ * slen)
        cD_update = soft_threshold(cD,lambda_ * slen)
        #逆小波变换
        y_update = pywt.idwt2((cA,(cH_update,cV_update,cD_update)),'db4')

        y_update = y_update[:original_shape[0], :original_shape[1]]
        # FISTA迭代更新
        x_new = y_update - slen * res
        t_new = (1 + np.sqrt(1 + 4 * t**2))/2
        y_new = x_new + ((t - 1) / t_new) * (x_new - x)
        #归一化
        x_new = np.clip(x_new,0,1)
        y_new = np.clip(y_new,0,1)
        #计算残差判断收敛
        residual = np.linalg.norm(x_new - prev_x)/np.linalg.norm(prev_x)
        convergence.append(residual)
        if residual < tol:
            print(f"迭代{i+1}次后收敛")
            break
        x,y,t = x_new,y_new,t_new
        prev_x = np.copy(x)
    return x,convergence

if __name__ == "__main__":
    #计时初始化
    total_start = time.time()
    single_time_list = []

    for name in os.listdir(date):
        single_start = time.time()
        img_path = os.path.join(date,name)
        #读取灰度图像并归一化
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = img.astype(np.float32)/255.0
        base = os.path.splitext(name)[0]

        noisy_img, mask = generate_missing_pixels(img,ratio)
        save_name = f"{base}_noisy.png"
        save_path = os.path.join(save,save_name)
        cv2.imwrite(save_path, (noisy_img * 255).astype(np.uint8))
        
        #FISTA图像补全
        inpainted_img, convergence = fista(noisy_img,mask)
        inpainted_img = np.clip(inpainted_img,0,1)
        
        #保存补全后的图像
        inpainted_save_name = f"{base}_denoised.png"
        inpainted_save_path = os.path.join(result,inpainted_save_name)
        cv2.imwrite(inpainted_save_path,(inpainted_img * 255).astype(np.uint8))
        
        #计算PSNR SSIM
        psnr_value = psnr(img,inpainted_img,data_range=1.0)
        ssim_value = ssim(img,inpainted_img,data_range=1.0)

        single_end = time.time()
        single_cost = round(single_end - single_start,4)
        single_time_list.append(single_cost)

        print(f"{name}")
        print(f"---PSNR: {psnr_value:.2f} dB --- SSIM: {ssim_value:.4f}")
        print(f"---迭代次数: {len(convergence)}   最终收敛残差: {convergence[-1]:.6f}")
        print(f"---处理时长: {single_cost:.4f} 秒\n")

    total_end = time.time()
    total_cost = round(total_end - total_start,4)
    avg_cost = round(np.mean(single_time_list),4) if single_time_list else 0.0

    print(f"总运行时长: {total_cost:.4f} 秒")
    print(f"单张图片平均处理时长: {avg_cost:.4f} 秒")
