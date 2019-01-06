import tensorflow as tf
from object_detection.utils import dataset_util
import pickle
import glob
from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageFilter
import numpy as np
import io

def load_obj(name ):
    load_dir  = "path/to/img"
    with open(load_dir + name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='bytes')


def create_all_batches():
    images = "path/to/img"
    image_names = []
    for filename in glob.glob(images+'/*.png'):
        image_names.append(filename)

    name = "labels"
    list_of_labels = load_obj(name) 
    images = []
    labels = [] 
    for img in image_names:
        key = img.split("/")[-1]
        img_tmp = Image.open(img)
        labls = list_of_labels[key][1]
        images.append(img)
        labels.append([np.float32(labls[1]), np.float32(labls[2]), np.float32(labls[3]), np.float32(labls[4])])
        img_tmp.close()

    images = np.asarray(images)
    labels = np.asarray(labels)
    train_data = []
    for im, labl in zip(images, labels):
      train_data.append((im,labl))

    return train_data

def create_tf_example(label_and_data_info, nr):
  height = 128 # Image height
  width = 128 # Image width
  filename = bytes(nr) # Filename of the image. Empty if image is not from file

  img_tmp = Image.open(label_and_data_info[0])
  imgByteArr = io.BytesIO()
  img_tmp.save(imgByteArr, format='PNG')
  img_tmp.close()
  encoded_image_data = imgByteArr.getvalue() # Encoded image bytes
  image_format = b'png'

  labels = label_and_data_info[1]
  # Normalize coord
  xmin = labels[0]/width
  xmax = (labels[0]+labels[2])/width # min + width
  ymin = labels[1]/height
  ymax = (labels[1] + labels[3])/height # min + height

  xmins = [xmin] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [xmax] 
  ymins = [ymin] 
  ymaxs = [ymax] 
  classes_text = [b'text'] # List of string class name of bounding box (1 per box)
  classes = [1] # List of integer class id of bounding box (1 per box)
  
  tf_label_and_data = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_label_and_data


# Separate for eval and train evaluation.record evaluation.record
flags = tf.app.flags
flags.DEFINE_string('output_path', '"path/to/save/tf_rec_img"', 'tfrec')
FLAGS = flags.FLAGS


def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  all_data_and_label_info = create_all_batches()

  i = 0
  for data_and_label_info in all_data_and_label_info:
    tf_example = create_tf_example(data_and_label_info, i)
    writer.write(tf_example.SerializeToString())
    i += 1

  writer.close()

if __name__ == '__main__':
  tf.app.run()


