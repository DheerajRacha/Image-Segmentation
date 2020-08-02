import os
import cv2
import numpy as np
import tensorflow as tf
from arguments import flags
from DataLoader import DataLoader
from models.DeepLabV3 import build_deeplabv3
slim = tf.contrib.slim


args = flags()

image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="InputImage")
annotation = tf.placeholder(tf.int32, shape=[None, None, None, args.num_class], name="Annotation")

output_logits, _ = build_deeplabv3(inputs=image, num_classes=args.num_class, pretrained_dir=args.resnet_ckpt)

checkpoint_path = os.path.join(args.ckpt_dir, 'latest_checkpoints/checkpoints.ckpt')
weights_restored_from_file = slim.get_variables_to_restore()
init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, weights_restored_from_file)
sess = tf.Session()
init_fn(sess)

colour_codes = np.array([[255, 255, 255], [0, 0, 0]], np.uint8)
data_loader = DataLoader(args.preds, "inference")
load_batch = data_loader.load_data(1)

for batch in range(data_loader.dataset_size):
    input_image, image_name = next(load_batch)

    seg_map = sess.run(annotation, feed_dict={image: input_image})
    seg_map = np.array(seg_map[0, :, :, :])
    seg_map = np.argmax(seg_map, axis=-1)
    seg_map = colour_codes[seg_map.astype(int)]

    save_name = os.path.join(args.output_dir, os.path.basename(image_name) + "_ann.jpeg")
    cv2.imwrite(save_name, cv2.cvtColor(np.uint8(seg_map), cv2.COLOR_RGB2BGR))
