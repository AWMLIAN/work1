import numpy as np
import cv2
import sys


# 从文本文件中读取点
def readPoints(file):
    # 创建数组存储点
    points = []
    with open(file) as file:
        for line in file:
            x, y = line.split()
            points.append((int(x), int(y)))
    return points


# 将使用srcTri和dstTri计算的仿射变换应用于src并返回结果图像。
def applyAffineTransform(src, srcTri, dstTri, size):
    # 给定一对三角形，找到仿射变换
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # 将仿射变换应用于src图片
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Warp和alpha将img1和img2的三角形区域混合到img中
def morphTriangle(img1, img2, img, t1, t2, t, alpha):
    # 找到每个三角形区域的包络矩形
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # 各个矩形左上角的偏移点
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # 填充三角形来获得掩码
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1, 0), 16, 0)

    # 将warpImage应用于小矩形块
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha混合矩形补丁
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # 将矩形块的三角形区域复制到输出图像
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect * mask


if __name__ == '__main__':
    filename1 = 'dilireba.jpg'
    filename2 = 'yangmi.jpg'

    # 两幅图的融合的比率，范围0到1
    alpha = 0.5

    # 读取图片
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)

    # 将矩阵转换为浮点数据
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    # 读取相关点
    points1 = readPoints(filename1 + '.txt')
    points2 = readPoints(filename2 + '.txt')
    points = []

    # 计算加权平均点坐标
    for i in range(0, len(points1)):
        x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
        y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
        points.append((x, y))

    # 为最后的输出分配空间
    imgMorph = np.zeros(img1.shape, dtype=img1.dtype)
    with open("tri.txt") as file:
        for line in file:
            x, y, z = line.split()

            x = int(x)
            y = int(y)
            z = int(z)

            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            t = [points[x], points[y], points[z]]

            # 一次合成一个三角形
            morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)

    # 输出结果
    cv2.imshow("Morphed Face.jpg", np.uint8(imgMorph))
    cv2.imwrite("video/001.jpg", np.uint8(imgMorph))
    cv2.waitKey(0)
