from PIL import Image
import numpy as np
import tensorflow as tf
import os

# SaveAsTFRecord
def SaveAsTFRecord(img_raw, label):
	img_raw = img_raw.tobytes()
	example = tf.train.Example(
		features=tf.train.Features(
			feature={
			"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
			'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
		}))
	writer.write(example.SerializeToString())



if __name__ == '__main__':
    n = 0
    inputDir_root = r'F:\SunanImageData\resize_augs'
    outputDir_root = r'F:\SunanImageData\tfrecordData'
    output_file = outputDir_root + "\\" + "atrain.tfrecords"
    # output_file = outputDir_root + "\\" + "stest.tfrecords"

    global writer
    writer = tf.python_io.TFRecordWriter(output_file)

    class_list = os.listdir(inputDir_root)

    for class_name in class_list:
        label = int(class_name)
        datas_folder = inputDir_root + '\\' + class_name
        images = os.listdir(datas_folder)
        for image in images:
            imagePath = datas_folder + '\\' + image

            # 将图像读成一列
            imgVectorArr = []
            img = np.array(Image.open(imagePath))
            for dim in range(3):
                temp = np.reshape(img[:, :, dim], (-1, 1)).squeeze()
                imgVectorArr.append(temp)

            imgVector = np.asarray(imgVectorArr)
            SaveAsTFRecord(imgVector, label)

            n += 1
            print("已处理：" + imagePath)

    print("已完成！共处理了 %s 张图片" % (n))


