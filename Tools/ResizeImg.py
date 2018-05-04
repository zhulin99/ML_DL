from PIL import Image
import os.path
import glob
import time


def ResizeImg(imageFile, outDir, width=128, height=128):
    img = Image.open(imageFile)
    try:
        new_img = img.resize((width, height), Image.BILINEAR)
        new_img.save(os.path.join(outDir, os.path.basename(imageFile)))
    except Exception as e:
        print(e)

if __name__ == '__main__':
    start = time.time()
    i = 0

    imageDir = "F:\\imagefile\\data\\p80s\\*.jpg"
    outDir = "F:\\imagefile\\result\\p80s"
    for imageFile in glob.glob(imageDir):
        ResizeImg(imageFile, outDir)
        i += 1
        print ('正在处理第 %s 张图片' % (i))

    c = time.time() - start
    print('程序运行耗时:%0.2f' % (c))
    print('总共处理了 %s 张图片' % (i))