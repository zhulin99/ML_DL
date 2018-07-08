from PIL import Image
import os.path
import time


def ResizeImg(imgpath, outpath, width=227, height=227):
    if os.path.isdir(outpath):
        pass
    else:
        os.mkdir(outpath)

    i = 0
    imageNames = os.listdir(imgpath);
    for imageName in imageNames:
        imagePath = imgpath + '\\' + imageName
        img = Image.open(imagePath)
        try:
            new_img = img.resize((width, height), Image.BILINEAR)
            new_img.save(os.path.join(outpath, os.path.basename(imageName)))
        except Exception as e:
            print(e)

        i += 1
        print('正在处理第 %s 张图片' % (i))
    print('总共处理了 %s 张图片' % (i))

if __name__ == '__main__':
    start = time.time()

    imagepathRoot = r"F:\SunanImageData\source"
    outpathRoot = r"F:\SunanImageData\resize_source"

    classFolders = os.listdir(imagepathRoot)
    for classFolderName in classFolders:
        labelName = int(classFolderName)
        datas_folder = imagepathRoot + '\\' + classFolderName
        outFolderPath = outpathRoot + '\\' + classFolderName
        ResizeImg(datas_folder, outFolderPath)

    c = time.time() - start
    print('程序运行耗时:%0.2f' % (c))
