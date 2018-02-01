import logging
import os

import cv2
import numpy as np
import tensorflow as tf
from vgg import vgg_16

import dataset_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')

FLAGS = flags.FLAGS

classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']
# RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [
                128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]


cm2lbl = np.zeros(256**3)
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def image2label(im):
    data = im.astype('int32')
    # cv2.imread. default channel layout is BGR
    idx = (data[:, :, 2] * 256 + data[:, :, 1]) * 256 + data[:, :, 0]
    return np.array(cm2lbl[idx])


def dict_to_tf_example(data, label):
    with open(data, 'rb') as inf:
        encoded_data = inf.read()
    img_label = cv2.imread(label)
    img_mask = image2label(img_label)
    encoded_label = img_mask.astype(np.uint8).tobytes()
#     print(type(encoded_label))

    height, width = img_label.shape[0], img_label.shape[1]
    if height < vgg_16.default_image_size or width < vgg_16.default_image_size:
        
        return None

    # Your code here, fill the dict
    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(data.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_data),
        'image/label': dataset_util.bytes_feature(encoded_label),
        'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def create_tf_record(output_filename, file_pars):
    print('this is create tf record file function!')
    """Creates a TFRecord file from examples.

      Args:
        output_filename: Path to where output file is saved.
        label_map_dict: The label map dictionary.
        annotations_dir: Directory where annotation files are stored.
        image_dir: Directory where image files are stored.
        examples: Examples to parse and save to tf record.
        faces_only: If True, generates bounding boxes for pet faces.  Otherwise
          generates bounding boxes (as well as segmentations for full pet bodies).
     """
    print('-------1-------')
    writer = tf.python_io.TFRecordWriter(output_filename)
    print('-------2-------')
    print(type(file_pars))
    for example in file_pars:
        data = example[0]
        label = example[1]
        try:
            tf_example = dict_to_tf_example(data,label)
            if tf_example:
                writer.write(tf_example.SerializeToString())
        except ValueError:
            print('Invalid example: %s, ignoring.', data)
        
    print('-------4------')
#         print('--------------=========----------------')
#         data = example[0]
#         label = example[1]
#         print('###############data : %s,label : %s##############'%(data,label))
#         try:
#             tf_example = dict_to_tf_example(
#                data, label)
#             writer.write(tf_example.SerializeToString())
#         except ValueError:
#            logging.warning('Invalid example: %s, ignoring.', data)

    writer.close()


def read_images_names(root, train=True):
    txt_fname = os.path.join(root, 'ImageSets/Segmentation/', 'train.txt' if train else 'val.txt')

    with open(txt_fname, 'r') as f:
        images = f.read().split()

    data = []
    label = []
    for fname in images:
        data.append('%s/JPEGImages/%s.jpg' % (root, fname))
        label.append('%s/SegmentationClass/%s.png' % (root, fname))
    return zip(data, label)


def main(_):
    logging.info('Prepare dataset file names')

    train_output_path = os.path.join(FLAGS.output_dir, 'fcn_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'fcn_val.record')

    train_files = read_images_names(FLAGS.data_dir, True)
    val_files = read_images_names(FLAGS.data_dir, False)
    train_files = list(train_files)
    print('******************start*****************************')
#     print(train_files1[0])
    create_tf_record(train_output_path, train_files)
    create_tf_record(val_output_path, val_files)
    print('******************end*****************************')


if __name__ == '__main__':
    tf.app.run()
