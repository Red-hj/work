# work
一、功能：
对灰度图像添加高斯 + 椒盐混合噪声，使用 BM3D 算法去噪，分别含噪和去噪图像，并计算 PSNR、SSIM
二、参数说明
三、核心参数
| 参数               | 说明                     |
|--------------------|--------------------------|
| DATASET_PATH       | 原始灰度图像路径         |
| NOISY_SAVE_PATH    | 含噪图像保存路径         |
| RESULT_SAVE_PATH   | 去噪结果保存路径         |
| GAUSSIAN_SIGMA     | 高斯噪声标准差（σ=25）|
| SALT_PEPPER_AMOUNT | 椒盐噪声比例（2%）|
