from PIL import Image
import numpy as np
import h5py
import glob

# 创建图片的hdf5数据
def CreateH5(imgsData, outDir):
    list_classes = np.array(['80','90'], dtype = 'S7')
    f = h5py.File(outDir, 'w')
    f['list_classes'] = list_classes
    f['test_set_x'] = imgsData
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

def GetImgData(imageDir):
    imgCount, rows, cols, dims = GetImgInfo(imageDir)
    x_n = rows * cols * dims
    imgs = np.zeros((x_n, imgCount), dtype=int)
    i = 0
    for imageFile in glob.glob(imageDir):
        imgVector = []
        img = np.array(Image.open(imageFile))
        for dim in range(dims):
            temp = np.reshape(img[:, :, dim], (-1,1)).squeeze()
            imgVector += temp.tolist()
        imgs[:, i] = np.asarray(imgVector)
        i += 1
    return imgs


if __name__ == '__main__':
    imageDir = "F:\\imagefile\\test\\*.jpg"
    outDir = "F:\\imagefile\\hdf5\\aaa.h5"
    imgData = GetImgData(imageDir)
    CreateH5(imgData,outDir)
    print("finish")

