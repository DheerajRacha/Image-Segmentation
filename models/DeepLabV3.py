import os
import tensorflow as tf
from models import resnet_v2
from tensorflow.contrib import slim


def Upsampling(inputs,feature_map_shape):
    return tf.image.resize_bilinear(inputs, size=feature_map_shape)


def AtrousSpatialPyramidPoolingModule(inputs, depth=256):
    feature_map_size = tf.shape(inputs)

    # Global average pooling
    image_features = tf.reduce_mean(inputs, [1, 2], keep_dims=True)
    image_features = slim.conv2d(image_features, depth, [1, 1], activation_fn=None)
    image_features = tf.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))

    atrous_pool_block_1 = slim.conv2d(inputs, depth, [1, 1], activation_fn=None)
    atrous_pool_block_6 = slim.conv2d(inputs, depth, [3, 3], rate=6, activation_fn=None)
    atrous_pool_block_12 = slim.conv2d(inputs, depth, [3, 3], rate=12, activation_fn=None)
    atrous_pool_block_18 = slim.conv2d(inputs, depth, [3, 3], rate=18, activation_fn=None)

    net = tf.concat((image_features, atrous_pool_block_1, atrous_pool_block_6, atrous_pool_block_12, atrous_pool_block_18), axis=3)
    net = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_output", activation_fn=None)

    return net


def build_deeplabv3(inputs, num_classes, pretrained_dir, preset_model='DeepLabV3-Res101', weight_decay=1e-5, is_training=True):

    if preset_model == 'DeepLabV3-Res50':
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v2.resnet_v2_50(inputs, is_training=is_training, scope='resnet_v2_50')
            resnet_scope='resnet_v2_50'
            # DeepLabV3 requires pre-trained ResNet weights
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_dir, 'resnet_v2_50.ckpt'), slim.get_model_variables('resnet_v2_50'))
    elif preset_model == 'DeepLabV3-Res101':
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v2.resnet_v2_101(inputs, is_training=is_training, scope='resnet_v2_101')
            resnet_scope='resnet_v2_101'
            # DeepLabV3 requires pre-trained ResNet weights
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_dir, 'resnet_v2_101.ckpt'), slim.get_model_variables('resnet_v2_101'))
    elif preset_model == 'DeepLabV3-Res152':
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v2.resnet_v2_152(inputs, is_training=is_training, scope='resnet_v2_152')
            resnet_scope='resnet_v2_152'
            # DeepLabV3 requires pre-trained ResNet weights
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_dir, 'resnet_v2_152.ckpt'), slim.get_model_variables('resnet_v2_152'))
    else:
        raise ValueError("Unsupported ResNet model '%s'. This function only supports ResNet 50, ResNet 101, and ResNet 152" % (preset_model))

    label_size = tf.shape(inputs)[1:3]

    net = AtrousSpatialPyramidPoolingModule(end_points['pool4'])
    net = Upsampling(net, label_size)
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')

    return net, init_fn


def mean_image_subtraction(inputs, means=[123.68, 116.78, 103.94]):
    inputs=tf.to_float(inputs)
    num_channels = inputs.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=inputs)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)
