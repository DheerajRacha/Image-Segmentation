import os
import sys
import csv
import numpy as np
import tensorflow as tf
from arguments import flags
from DataLoader import DataLoader
from models.DeepLabV3 import build_deeplabv3
from utils import evaluate_segmentation


def train(args):
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="InputImage")
    annotation = tf.placeholder(tf.int32, shape=[None, None, None, args.num_class], name="Annotation")

    output_logits, init_fn = build_deeplabv3(inputs=image,
                                             num_classes=args.num_class,
                                             pretrained_dir=args.resnet_ckpt)

    loss_fn = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(logits=output_logits,
                                                                      labels=annotation, name="Loss")))
    theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss_fn, theta), 5)
    optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=0.5, beta2=0.99)
    train_op = optimizer.apply_gradients(zip(grads, theta))

    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=10)
    init_fn(sess)

    sess.run(tf.global_variables_initializer())

    for epoch in range(args.num_epochs):
        print("Epoch: {}".format(epoch))

        ''' Train for one epoch '''
        data_loader = DataLoader(args.train_dir, "train")
        load_batch = data_loader.load_data(args.batch_size)
        num_batches = int(data_loader.dataset_size / args.batch_size)

        for batch in range(num_batches):
            x_batch, y_batch = next(load_batch)
            feed_dict = {image: x_batch, annotation: y_batch}
            sess.run([train_op], feed_dict=feed_dict)

            if batch % 1000 == 0:
                sys.stdout.write("\tBatch: " + str(batch) + "\r")
                sys.stdout.flush()

        if epoch == 0:
            eval_file = open("evaluation.txt", "w")
            eval_file.write("Epoch, avg_accuracy, precision, recall, f1 score, mean iou\n")
            writer = csv.writer(eval_file)
        else:
            eval_file = open("evaluation.txt", "a")
            writer = csv.writer(eval_file)

        ''' Evaluate on Training Dataset '''
        loss_train = 0
        load_batch = data_loader.load_data(args.batch_size)
        print("\n\tLoss on Train Dataset")
        for batch in range(num_batches):
            x_batch, y_batch = next(load_batch)
            feed_dict = {image: x_batch, annotation: y_batch}
            loss_train += sess.run(loss_fn, feed_dict=feed_dict)
            if batch % 1000 == 0:
                sys.stdout.write("\t\tBatch: %d\r" % batch)
                sys.stdout.flush()

        print("\tTrain Loss: " + str(loss_train))

        ''' Evaluate on Validation Dataset '''
        loss_val = 0
        scores_list, class_scores_list, precision_list, recall_list, f1_list, iou_list = [], [], [], [], [], []

        data_loader = DataLoader(args.val_dir, "train")
        load_batch = data_loader.load_data(args.batch_size)
        num_batches = int(data_loader.dataset_size / args.batch_size)
        print("\n\tLoss on Validation Dataset")
        for batch in range(num_batches):
            x_batch, y_batch = next(load_batch)
            feed_dict = {image: x_batch, annotation: y_batch}
            prediction_batch, loss = sess.run([annotation, loss_fn], feed_dict=feed_dict)
            loss_val += loss
            for pred, annot in zip(prediction_batch, y_batch):
                accuracy, class_accuracies, prec, rec, f1, iou = evaluate_segmentation(pred=pred,
                                                                                       label=annot,
                                                                                       num_classes=args.num_class)
                scores_list.append(accuracy)
                class_scores_list.append(class_accuracies)
                precision_list.append(prec)
                recall_list.append(rec)
                f1_list.append(f1)
                iou_list.append(iou)

            if batch % 100 == 0:
                sys.stdout.write("\t\tBatch: %d\r" % batch)
                sys.stdout.flush()

        avg_accuracy = np.mean(scores_list)
        avg_prec = np.mean(precision_list)
        avg_rec = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_iou = np.mean(iou_list)
        fields = [epoch, avg_accuracy, avg_prec, avg_rec, avg_f1, avg_iou]
        writer.writerow(fields)

        print("\tValidation Loss: " + str(loss_val))

        ''' Save Checkpoints for every 10 epochs '''
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(args.ckpt_dir, str(epoch))
            os.makedirs(checkpoint_path)
            checkpoint_path = os.path.join(checkpoint_path, "checkpoints.ckpt")
            saver.save(sess, checkpoint_path)

        fields = [loss_train, loss_val]
        if epoch == 0:
            with open("losses.txt", "w") as f:
                writer = csv.writer(f)
                writer.writerow(fields)
        else:
            with open("losses.txt", "a") as f:
                writer = csv.writer(f)
                writer.writerow(fields)
    latest = os.path.join(args.ckpt_dir, "latest_checkpoints")
    if not os.path.isdir(latest):
        os.makedirs(latest)
    saver.save(sess, latest)


if __name__ == "__main__":
    train(flags())
