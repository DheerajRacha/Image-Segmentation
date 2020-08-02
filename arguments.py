import tensorflow as tf


def flags():
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_integer('num_class', "2", "number of classes for segmentation")
    tf.flags.DEFINE_integer("image_size", "500", "batch size for training")
    tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
    tf.flags.DEFINE_integer('num_epochs', "300", "number of epochs")
    tf.flags.DEFINE_string("train_dir", "inria_dataset/train", "path of train image")
    tf.flags.DEFINE_string("val_dir", "inria_dataset/validation", "path of test image")
    tf.flags.DEFINE_float("learning_rate", "5e-5", "Learning rate for Adam Optimizer")
    tf.flags.DEFINE_float("dropout_rate", "0.3", "dropout rate")
    tf.flags.DEFINE_string('resnet_ckpt', "resnet_v2_101_2017_04_14", "path to checkpoints of resnetv2_101")
    tf.flags.DEFINE_string('ckpt_dir', "checkpoints", "Path to save checkpoints")
    tf.flags.DEFINE_string('preds', 'preds', 'path to input images for inference')
    tf.flags.DEFINE_string('output_dir', 'output', 'Path to save segmentation maps')
    return FLAGS
