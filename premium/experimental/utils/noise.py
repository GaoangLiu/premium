#!/usr/bin/env python
import matplotlib.pyplot as plt
import codefast as cf
import numpy as np
from typing import List, Union, Callable, Set, Dict, Tuple, Optional
import cv2


def _gaussian_noise(img, mean: float, sigma: float):
    '''
    此函数用将产生的高斯噪声加到图片上
    传入:
        img   :  原图
        mean  :  均值
        sigma :  标准差
    返回:
        gaussian_out : 噪声处理后的图片
        noise        : 对应的噪声
    '''
    # 将图片灰度标准化
    img = img / 255
    noise = np.random.normal(mean, sigma, img.shape)
    gaussian_out = img + noise
    gaussian_out = np.clip(gaussian_out, 0, 1)
    gaussian_out = np.uint8(gaussian_out*255)
    return gaussian_out, noise


def gaussian(image_path: str, out_path: str, sigma: float):
    """ Add gaussian noise to image and save it to out_path 
    """
    src = cv2.imread(image_path)
    gaussian_out, _ = _gaussian_noise(src, 0, sigma)
    cv2.imwrite(out_path, gaussian_out)


def add_gaussian_noise(image_path: str, out_path: str):
    src = cv2.imread(image_path)
    fig_out = plt.figure(figsize=(4, 2), dpi=370)  # figsize宽高比
    fig_noise = plt.figure(figsize=(4, 2), dpi=370)

    for i in range(0, 8):
        gaussian_out, noise = _gaussian_noise(src, 0, 0.03*i)

        ax_out = fig_out.add_subplot(i+241)
        ax_noise = fig_noise.add_subplot(i+241)
        ax_out.axis('off')
        ax_noise.axis('off')

        ax_out.set_title('$\sigma$ = '+str(0.03*i), loc='left',
                         fontsize=3, fontstyle='italic')
        ax_noise.set_title('$\sigma$ = '+str(0.03*i),
                           loc='left', fontsize=3, fontstyle='italic')

        ax_out.imshow(gaussian_out, cmap='gray')
        ax_noise.imshow((noise+1)/2, cmap='gray')

    fig_out.savefig(out_path)
    fig_noise.savefig('/tmp/gaussian_noise.png')
    plt.show()
