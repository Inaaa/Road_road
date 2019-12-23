import tensorflow as tf
import os
import matplotlib.pyplot as plt



os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# 函数的功能时将filename对应的图片文件读进来，并缩放到统一的大小
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_png(image_string)
  image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  image_resized.set_shape([28,28,3])
  with tf.Session() as sess:
      print('image_resized {}'.format(image_resized))
      print(image_resized.eval())
  return image_resized, label

# 图片文件的列表
filenames = tf.constant(["/home/chli/cc_code/road_segmentation/data/train/image/1.jpg",
                         "/home/chli/cc_code/road_segmentation/data/train/image/2.jpg"])
# label[i]就是图片filenames[i]的label
labels = tf.constant(["/home/chli/cc_code/road_segmentation/data/train/image/1.jpg",
                         "/home/chli/cc_code/road_segmentation/data/train/image/2.jpg"])

# 此时dataset中的一个元素是(filename, label)
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

# 此时dataset中的一个元素是(image_resized, label)
dataset = dataset.map(_parse_function)

# 此时dataset中的一个元素是(image_resized_batch, label_batch)
dataset = dataset.shuffle(buffer_size=100).batch(32).repeat(10)
print(dataset)