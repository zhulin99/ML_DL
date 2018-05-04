from PIL import Image
import numpy as np
import h5py
import glob

# 创建图片的hdf5数据
def CreateH5(imgsData, flagsData, outDir):
    list_classes = np.array(['80','90'], dtype = 'S7')
    f = h5py.File(outDir, 'w')
    f['list_classes'] = list_classes
    f['test_set_x'] = imgsData
    f['test_set_y'] = flagsData
    f.close()

# 获取图片库文件夹中图片信息
def GetImgInfo(imageDir):
    imgCount = 0
    for imageFile in glob.glob(imageDir):
        if imgCount == 0:
            img = np.array(Image.open(imageFile))
            rows, cols, dims = img.shape
        imgCount += 1
    return imgCount, rows, cols, dims

# 将图片数据转换为4维数组（mx128x128x3）
def GetImgData(imageDir):
    # 获取图片库文件夹中图片信息
    imgCount, rows, cols, dims = GetImgInfo(imageDir)
    # 定义图片数组与对应的标记数组
    imgs = np.zeros((imgCount, rows, cols, dims), dtype=int)
    flags = np.zeros(imgCount, dtype=int)
    i = 0
    for imageFile in glob.glob(imageDir):
        # 打开图像并转化为数字矩阵(128x128x3)
        img = np.array(Image.open(imageFile))
        imgs[i] = img
        # 获取图像类别标签
        flag = int(imageFile[-6:-4])
        if flag == 80:
            flags[i] = 0
        else:
            flags[i] = 1
        i += 1
    return imgs, flags

# 将图片数据转换为2维数组(128*128*3,m)即每一个图片对象是一个列向量
def GetImgData2(imageDir):
    imgCount, rows, cols, dims = GetImgInfo(imageDir)
    x_n = rows * cols * dims
    imgs = np.zeros((x_n, imgCount), dtype=int)
    flags = np.zeros(imgCount, dtype=int)
    i = 0
    for imageFile in glob.glob(imageDir):
        # 获取图片特征向量
        imgVector = []
        img = np.array(Image.open(imageFile))
        for dim in range(dims):
            temp = np.reshape(img[:, :, dim], (-1, 1)).squeeze()
            imgVector += temp.tolist()
        imgs[:, i] = np.asarray(imgVector)
        # 获取图像类别标签
        flag = int(imageFile[-6:-4])
        if flag == 80:
            flags[i] = 0
        else:
            flags[i] = 1
        i += 1
    return imgs, flags


if __name__ == '__main__':
    imageDir = "F:\\imagefile\\logisticdata\\test\\*.jpg"
    outDir = "F:\\imagefile\\hdf5\\test.h5"
    imgsData, flagsData = GetImgData(imageDir)
    CreateH5(imgsData, flagsData, outDir)
    print(imgsData.shape)
