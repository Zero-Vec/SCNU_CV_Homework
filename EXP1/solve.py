import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 设置图片路径
Al_path = os.path.join("image", "AI.jpg")
Fe_path = os.path.join("image", "Fe.jpg")
P_path = os.path.join("image", "P.jpg")
# print("图片形状:", al.shape)

# 读取三张图片
Al = cv2.imread(Al_path)
Fe = cv2.imread(Fe_path)
P = cv2.imread(P_path)

# RGB2HSV
def rgb2hsv(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    return [h, s, v]

# Plot image
def plot_3image(img: list):
    plt.figure(figsize=(15, 15))
    for i, img in enumerate(img):
        plt.subplot(1, 3, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

# Plot HSV image
def plot_hsv_image(img: list):
    cv2.imshow('H', img[0])
    cv2.imshow('S', img[1])
    cv2.imshow('V', img[2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 完成实验拓展要求，去除 logo 区域 和 黑白点
def remove_LogoAndPoints(split_img: list):
    h, s, v = split_img
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            # 去除黑色像素点
            if 0 <= h[i, j] <= 180 and 0 <= v[i, j] <= 46:
                h[i, j] = 0
                s[i, j] = 0
                v[i, j] = 0
            # 去除白色像素点
            elif 0 <= h[i, j] <= 180 and 221 <= v[i, j] <= 255 and 0 <= s[i, j] <= 30:
                h[i, j] = 0
                s[i, j] = 0
                v[i, j] = 0
            # 去除 logo 区域
            if i <= 70 and 180 <= v[i, j] <= 255:
                h[i, j] = 0
                s[i, j] = 0
                v[i, j] = 0
    return [h, s, v]

# 统计各化学元素的数目
def count_points(img: list):
    h, s, v = img
    cnt = 0
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if v[i, j] != 0:
                cnt += 1
    return cnt

images = [Al, Fe, P]
# plot_3image(images)

Al_hsv = rgb2hsv(Al)
Fe_hsv = rgb2hsv(Fe)
P_hsv = rgb2hsv(P)
# plot_hsv_image(rgb2hsv(Al))

# 预处理，去除 logo 以及 50nm 标识
pre_Al = remove_LogoAndPoints(Al_hsv)
pre_Fe = remove_LogoAndPoints(Fe_hsv)
pre_P = remove_LogoAndPoints(P_hsv)

# 统计各化学元素的数目
cnt_Al = count_points(pre_Al)
cnt_Fe = count_points(pre_Fe)
cnt_P = count_points(pre_P)

print("铝元素的点数为: ", cnt_Al)
print("铁元素的点数为: ", cnt_Fe)
print("磷元素的点数为: ", cnt_P)

# 求解重叠量，并将元素两两的重叠量画出来
def cnt2overlap(img1: list, img2: list):
    h1, s1, v1 = img1
    h2, s2, v2 = img2
    # 找到两者重叠的区域
    overlap_mask = (v1 != 0) & (v2 != 0)
    # 计算重叠量
    cnt = np.sum(overlap_mask)
    # 计算 h0, s0, v0
    h0 = np.zeros_like(h1, dtype=np.uint8)
    s0 = np.zeros_like(s1, dtype=np.uint8)
    v0 = np.zeros_like(v1, dtype=np.uint8)
    h0[overlap_mask] = np.clip(h1[overlap_mask] + h2[overlap_mask], 0, 255)
    s0[overlap_mask] = np.clip(s1[overlap_mask] + s2[overlap_mask], 0, 255)
    v0[overlap_mask] = np.clip(v1[overlap_mask] + v2[overlap_mask], 0, 255)
    return cnt, [h0, s0, v0]

# 同理，求三者的重叠量
def cnt3overlap(img1: list, img2: list, img3: list):
    h1, s1, v1 = img1
    h2, s2, v2 = img2
    h3, s3, v3 = img3
    # 找到三者重叠的区域
    overlap_mask = (v1 != 0) & (v2 != 0) & (v3 != 0)
    # 计算重叠量
    cnt = np.sum(overlap_mask)
    # 计算 h0, s0, v0
    h0 = np.zeros_like(h1, dtype=np.uint8)
    s0 = np.zeros_like(s1, dtype=np.uint8)
    v0 = np.zeros_like(v1, dtype=np.uint8)
    h0[overlap_mask] = np.clip(h1[overlap_mask] + h2[overlap_mask] + h3[overlap_mask], 0, 255)
    s0[overlap_mask] = np.clip(s1[overlap_mask] + s2[overlap_mask] + s3[overlap_mask], 0, 255)
    v0[overlap_mask] = np.clip(v1[overlap_mask] + v2[overlap_mask] + v3[overlap_mask], 0, 255)
    return cnt, [h0, s0, v0]

# 算两两重叠率
cnt_Al_and_Fe, img_Al_and_Fe = cnt2overlap(pre_Al, pre_Fe)
cnt_Al_and_P, img_Al_and_P = cnt2overlap(pre_Al, pre_P)
cnt_Fe_and_P, img_Fe_and_P = cnt2overlap(pre_Fe, pre_P)

img_Al_and_Fe = cv2.cvtColor(cv2.merge(img_Al_and_Fe), cv2.COLOR_HSV2BGR)
img_Al_and_P = cv2.cvtColor(cv2.merge(img_Al_and_P), cv2.COLOR_HSV2BGR)
img_Fe_and_P = cv2.cvtColor(cv2.merge(img_Fe_and_P), cv2.COLOR_HSV2BGR)

print("铝和铁的重叠量为: ", cnt_Al_and_Fe)
print("铝和磷的重叠量为: ", cnt_Al_and_P)
print("铁和磷的重叠量为: ", cnt_Fe_and_P)
# plot_3image([img_Al_and_Fe, img_Al_and_P, img_Fe_and_P])

# 算三者重叠率
cnt_all, img_all = cnt3overlap(pre_Al, pre_Fe, pre_P)
img_all = cv2.cvtColor(cv2.merge(img_all), cv2.COLOR_HSV2BGR)
print("铝、铁、磷三者的重叠量为: ", cnt_all)
cv2.imshow('All overlap', img_all)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存两两元素重叠的图片、保存三者元素重叠的图片
if not os.path.exists("image_output"):
    os.makedirs("image_output")
cv2.imwrite(os.path.join("image_output", "Al_and_Fe.jpg"), img_Al_and_Fe)
cv2.imwrite(os.path.join("image_output", "Al_and_P.jpg"), img_Al_and_P)
cv2.imwrite(os.path.join("image_output", "Fe_and_P.jpg"), img_Fe_and_P)
cv2.imwrite(os.path.join("image_output", "All_overlap.jpg"), img_all)
