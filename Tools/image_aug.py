# -*- coding:utf-8 -*-

from imgaug import augmenters as iaa
import cv2
import os


def image_aug(imgpath, outpath):
    if os.path.isdir(outpath):
        pass
    else:
        os.mkdir(outpath)

    seq = iaa.Sequential([
        iaa.Crop(percent=0.15),              # 随机裁剪原图像的10%
        iaa.Fliplr(1.0),                    # 图像做水平翻转
        iaa.GaussianBlur(sigma=(0, 3.0)),   # 随机从（0-3）对图像做高斯扰动
        # 对图像添加高斯噪声
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # 对图像做放射变换
        # iaa.Affine(
        #     scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        #     translate_percent={"x": (-0.1, 0.1), "y": (-0.12, 0)},
        #     rotate=(-10, 10),
        #     shear=(-8, 8),
        #     order=[0, 1],
        #     cval=(0, 255),
        # ),
        # 先将图片从RGB变换到HSV,然后将V值增加10,然后再变换回RGB。
        iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB", children=iaa.WithChannels(2, iaa.Add(10)))
    ])

    imglist = []
    imageNames = os.listdir(imgpath);
    for imageName in imageNames:
        # 处理数据
        imagePath = imgpath + '\\' + imageName
        imageData = cv2.imread(imagePath)
        imglist.append(imageData);

        images_aug = seq.augment_images(imglist)

        # 保存数据
        imageNameWithoutExt = imageName.split('.')[0]
        augImageName = imageNameWithoutExt + '_' + '.jpg'
        cv2.imwrite(outpath + '\\' + augImageName, images_aug[0])
        print("已完成：" + outpath + '\\' + augImageName)

        imglist.clear()


if __name__ == '__main__':
    imagepathRoot = r"F:\SunanImageData\source"
    outpathRoot = r"F:\SunanImageData\imgaugs"

    classFolders = os.listdir(imagepathRoot)
    for classFolderName in classFolders:
        labelName = int(classFolderName)
        datas_folder = imagepathRoot + '\\' + classFolderName
        outFolderPath = outpathRoot + '\\' + classFolderName
        image_aug(datas_folder, outFolderPath)

    print("已完成全部!")
