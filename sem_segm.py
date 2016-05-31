from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import scipy.misc
import scipy.io
import os.path
import load_data
import vgg19 as vgg
from utils import conv_layer
import time
import logging

from config import VGG_PATH, CHECKPOINT_DIR, SUMMARY_DIR, SAMPLE_IMAGE_PATH, LOG_FILE, \
    MAX_STEPS, BATCH_SIZE, ACCURACY_EVAL_TRAIN_PORTION, ACCURACY_EVAL_TEST_PORTION, \
    DROP_PROB, SAVE_AND_EVAL_EVERY, SUMMARY_EVERY, MAX_HEIGHT, MAX_WIDTH, MEAN, \
    EXAMPLE_IMAGE_ID, TRAIN


def get_inference(images_ph, dropout_keep_prob_ph):
    #subtract average image
    with tf.variable_scope('centering') as scope:
        mean = tf.constant(vgg.average_image, dtype=tf.float32, name='avg_image')
        images_ph = tf.sub(images_ph, mean, name='subtract_avg')

    #get layers from vgg19
    vgg_layers = vgg.get_VGG_layers(images_ph, dropout_keep_prob_ph, train_fc_layers=True)

    #################################################
    ### Add more layers for semantic segmentation ###
    #################################################

    # convolution on top of pool4 to 21 chammenls (to make coarse predictions)
    with tf.variable_scope('conv9') as scope:
        conv9 = conv_layer(vgg_layers['pool4'], 21, 1, 'conv9')

    # convolution on top of conv7 (fc7) to 21 chammenls (to make coarse predictions)
    with tf.variable_scope('conv8') as scope:
        conv8 = conv_layer(vgg_layers['dropout2'], 21, 1, 'conv8')

    # 2x upsampling from last layer
    with tf.variable_scope('deconv1') as scope:
        shape = tf.shape(conv8)
        out_shape = tf.pack([shape[0], shape[1]*2, shape[2]*2, 21])
        weights = tf.Variable(tf.truncated_normal(mean=MEAN, stddev=0.1, shape=(4, 4, 21, 21)), name='weights')
        deconv1 = tf.nn.conv2d_transpose( value=conv8,
                                          filter=weights,
                                          output_shape=out_shape,
                                          strides=(1, 2, 2, 1),
                                          padding='SAME',
                                          name='deconv1')

        # slice 2x upsampled tensor in the last layer to fit pool4
        shape = tf.shape(conv9)
        size = tf.pack([-1, shape[1], shape[2], -1])
        deconv1 = tf.slice(deconv1, begin=[0,0,0,0], size=size, name="deconv1_slice")

    # combine preductions from last layer and pool4
    with tf.variable_scope('combined_pred') as scope:
        combined_pred = tf.add(deconv1, conv9, name="combined_pred")

    # 16x upsampling
    with tf.variable_scope('deconv2') as scope:
        shape = tf.shape(combined_pred)
        out_shape = tf.pack([shape[0], shape[1]*16, shape[2]*16, 21])
        weights = tf.Variable(tf.truncated_normal(mean=MEAN, stddev=0.1, shape=(32, 32, 21, 21)), name='weights')
        deconv2 = tf.nn.conv2d_transpose(value=combined_pred,
                                          filter=weights,
                                          output_shape=out_shape,
                                          strides=(1, 16, 16, 1),
                                          padding='SAME',
                                          name='deconv2')

        # slice upsampled tensor to original shape
        orig_shape = tf.shape(images_ph)
        size = tf.pack([-1, orig_shape[1], orig_shape[2], -1])
        logits = tf.slice(deconv2, begin=[0,0,0,0], size=size, name='logits')

    return logits


def get_predictions(logits):
    #Pixel-wise softmax
    with tf.variable_scope('softmax') as scope:
        pred_shape = tf.shape(logits)
        logits = tf.reshape(logits, [-1, 21])
        y_sotfmax = tf.nn.softmax(logits, name='softmax1d')
        y_sotfmax = tf.reshape(y_sotfmax, pred_shape, name='softmax')
    return y_sotfmax


def get_loss(logits, labels):
    with tf.variable_scope('loss') as scope:
        #reshape
        logits = tf.reshape(logits, [-1, 21], name='logits2d')
        labels = tf.reshape(labels, [-1], name='labels1d')
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
    return loss


def get_training(loss, global_step):
    learning_rate = tf.train.exponential_decay(1e-4, global_step, 500, 0.96, staircase=False)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op, learning_rate


def get_eval(logits, labels):
    with tf.variable_scope('loss2') as scope:
        logits = tf.reshape(logits, [-1, 21], name='logits2d')
        labels = tf.reshape(labels, [-1], name='labels1d')
        y_sotfmax = tf.nn.softmax(logits, name='softmax1d')
        predictions = tf.argmax(y_sotfmax, 1)
        correct_pred = tf.to_float(tf.equal(labels, predictions))
        ones = tf.ones_like(labels)
        eval_count = tf.to_float(tf.unsorted_segment_sum(ones, labels, 21))
        eval_correct = tf.to_float(tf.unsorted_segment_sum(correct_pred, labels, 21))
    return eval_count, eval_correct


def do_eval(sess,
            eval_count,
            eval_correct,
            images_ph,
            labels_ph,
            dropout_keep_prob_ph,
            data_set,
            sample=0.25,
            it=None):

    if it is not None:
        steps = it
    else:
        steps = int(sample * data_set.num_examples) // BATCH_SIZE

    total_count = np.zeros(shape=(21,))
    total_correct = np.zeros(shape=(21,))

    for step in xrange(steps):
        batch = data_set.next_batch(BATCH_SIZE)
        feed_dict = {images_ph: batch[0], labels_ph: batch[1], dropout_keep_prob_ph: 1.0}
        count, correct_pred = sess.run([eval_count, eval_correct], feed_dict=feed_dict)
        total_count = total_count + count
        total_correct = total_correct + correct_pred

    nonzero = np.nonzero(total_count)
    total_count = total_count[nonzero]
    total_correct = total_correct[nonzero]
    mean_class_accuracy = total_correct/total_count
    mean_accuracy = np.mean(mean_class_accuracy)
    true_pixels = np.sum(total_correct)
    num_pixels = np.sum(total_count)
    pixel_accuracy = true_pixels/num_pixels
    return pixel_accuracy, mean_accuracy


def get_best_class_prediction(predictions):
    with tf.variable_scope('image_eval') as scope:
        pred_se = tf.argmax(predictions, 3)
    return pred_se


def normalized_loss(loss, images_ph):
    '''Max size 20x320x320x3'''
    with tf.variable_scope('normalized_loss') as scope:
        max_pixels_in_batch = tf.constant(BATCH_SIZE * MAX_HEIGHT * MAX_WIDTH * 3, dtype=tf.float32)
        shape = tf.shape(images_ph)
        pixels_in_batch = tf.to_float(shape[0] * shape[1] * shape[2] * shape[3])
        norm = tf.div(pixels_in_batch, max_pixels_in_batch)
        norm_loss = tf.mul(loss, norm)
    return norm_loss


def main():
    #logger
    logger = get_logger()
    logger.info('New run')

    #placeholders
    images_ph = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='images_ph')
    labels_ph = tf.placeholder(tf.int64, shape=(None, None, None), name='labels_ph')
    dropout_keep_prob_ph = tf.placeholder(tf.float32)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    logits = get_inference(images_ph, dropout_keep_prob_ph)
    predictions = get_predictions(logits)
    loss = get_loss(logits, labels_ph)
    norm_loss = normalized_loss(loss, images_ph)
    train_op, learning_rate = get_training(loss, global_step)
    image_eval = get_best_class_prediction(predictions)
    eval_count, eval_correct = get_eval(logits, labels_ph)

    #SUMMARIES
    loss_summary = tf.scalar_summary('loss/loss', loss)
    norm_loss_summary = tf.scalar_summary('loss/normalized_loss', norm_loss)
    learning_rate_summary = tf.scalar_summary('learning_rate', learning_rate)
    accuracy_pixel_ph = tf.placeholder(tf.float32, shape=(), name='accuracy_pixel_ph')
    accuracy_pixel_summary = tf.scalar_summary("accuracy/pixel_accuracy", accuracy_pixel_ph)
    accuracy_mean_ph = tf.placeholder(tf.float32, shape=(), name='accuracy_mean_ph')
    accuracy_mean_summary = tf.scalar_summary("accuracy/mean_accuracy", accuracy_mean_ph)

    summary_op_train = tf.merge_summary([loss_summary, norm_loss_summary, learning_rate_summary])
    summary_op_test = tf.merge_summary([loss_summary, norm_loss_summary])
    
    saver = tf.train.Saver(max_to_keep=3)
    
    with tf.Session() as sess:
        # Restore from checkpoint
        ckpt_path = tf.train.latest_checkpoint(CHECKPOINT_DIR)
        if ckpt_path is not None:
            saver.restore(sess, ckpt_path)
            msg = "restored from " + ckpt_path
            print(msg)
            logger.info(msg)
        elif os.path.isfile(VGG_PATH):
            init_ops = vgg.get_initialize_op_for_VGG()
            init_ops.append(tf.initialize_all_variables())
            sess.run(init_ops)
            saver.save(sess, CHECKPOINT_DIR + "/semantic_segmentation", global_step=0)
            msg = "initialized from .mat and saved as checkpoint"
            print(msg)
            logger.info(msg)
        else:
            sess.run(tf.initialize_all_variables())
            msg = "initialized with defaults"
            print(msg)
            logger.info(msg)

        train_dataset, test_dataset = load_data.get_datasets()
        
        gs = global_step.eval()
        
        if TRAIN:
            # Instantiate a SummaryWriter to output summaries and the Graph.
            summary_writer_train = tf.train.SummaryWriter(SUMMARY_DIR + '/train', sess.graph)
            summary_writer_test = tf.train.SummaryWriter(SUMMARY_DIR + '/test')

            # TRAINING
            
            for step in xrange(MAX_STEPS):
                start_time = time.time()
                gs = global_step.eval()

                batch = train_dataset.next_batch(BATCH_SIZE)
                feed_dict = {images_ph: batch[0],
                             labels_ph: batch[1],
                             dropout_keep_prob_ph: DROP_PROB}

                sess.run(train_op, feed_dict=feed_dict)

                duration = time.time() - start_time

                # Print status to stdout and log
                if gs != step:
                    msg = 'Global step %d (step = %d); %.3f sec' % (gs, step, duration)
                else:
                    msg = 'Global step %d; %.3f sec' % (gs, duration)
                print(msg)
                logger.info(msg)

                # Write the summaries and print an overview fairly often.
                if (gs + 1) % SUMMARY_EVERY == 0:
                    test_batch = train_dataset.next_batch(BATCH_SIZE)
                    feed_dict2 = {images_ph: test_batch[0],
                                  labels_ph: test_batch[1],
                                  dropout_keep_prob_ph: DROP_PROB}

                    # Update the events file.
                    start_time = time.time()

                    summary_str_train = sess.run(summary_op_train, feed_dict=feed_dict)
                    summary_writer_train.add_summary(summary_str_train, gs)
                    summary_writer_train.flush()

                    summary_str_test = sess.run(summary_op_test, feed_dict=feed_dict2)
                    summary_writer_test.add_summary(summary_str_test, gs)
                    summary_writer_test.flush()

                    duration = time.time() - start_time
                    msg = 'Summary saved (%.3f)'%(duration)
                    print(msg)
                    logger.info(msg)

                #evaluate, save model, save example image 
                if (gs + 1) % SAVE_AND_EVAL_EVERY == 0 or (step + 1) == MAX_STEPS:
                    #evaluate pixel accuracy
                    eval_train_dataset, eval_test_dataset = load_data.get_datasets()

                    start_time = time.time()

                    train_pixel_accuracy, train_mean_accuracy = do_eval(sess, eval_count, eval_correct, 
                                            images_ph, labels_ph, dropout_keep_prob_ph,
                                            eval_train_dataset, ACCURACY_EVAL_TRAIN_PORTION) 
                    duration = time.time() - start_time
                    msg = 'train_pixel_acc = %.4f, train_mean_acc = %.4f  (%.3f)' % (train_pixel_accuracy, train_mean_accuracy, duration)
                    print(msg)
                    logger.info(msg)

                    start_time = time.time()

                    test_pixel_accuracy, test_mean_accuracy = do_eval(sess, eval_count, eval_correct,
                                            images_ph, labels_ph, dropout_keep_prob_ph,
                                            eval_test_dataset, ACCURACY_EVAL_TEST_PORTION)
                    duration = time.time() - start_time

                    msg = 'test_pixel_acc = %.4f, test_mean_acc = %.4f (%.3f)' % (test_pixel_accuracy, test_mean_accuracy, duration)
                    print(msg)
                    logger.info(msg)

                    train_pixel_acc_str = sess.run(accuracy_pixel_summary, feed_dict={accuracy_pixel_ph: train_pixel_accuracy})
                    summary_writer_train.add_summary(train_pixel_acc_str, gs)

                    train_mean_acc_str = sess.run(accuracy_mean_summary, feed_dict={accuracy_mean_ph: train_mean_accuracy})
                    summary_writer_train.add_summary(train_mean_acc_str, gs)
                    summary_writer_train.flush()

                    test_pixel_acc_str = sess.run(accuracy_pixel_summary, feed_dict={accuracy_pixel_ph: test_pixel_accuracy})
                    summary_writer_test.add_summary(test_pixel_acc_str, gs)
                    test_mean_acc_str = sess.run(accuracy_mean_summary, feed_dict={accuracy_mean_ph: test_mean_accuracy})
                    summary_writer_test.add_summary(test_mean_acc_str, gs)
                    summary_writer_test.flush()

                    #compute and save sample image segmentation
                    im, se = load_data.load_one_image(EXAMPLE_IMAGE_ID)
                    one_image_pred = sess.run(image_eval, feed_dict={images_ph: im, labels_ph: se, dropout_keep_prob_ph: 1.0})
                    load_data.save_image(EXAMPLE_IMAGE_ID, one_image_pred, gs, SAMPLE_IMAGE_PATH)
                    msg = "example image saved"
                    print(msg)
                    logger.info(msg)

                    #save a checkpoint
                    saver.save(sess, CHECKPOINT_DIR + "/semantic_segmentation", global_step=gs)

            summary_writer_train.close()
            summary_writer_test.close()
        else:
            #compute and save sample image segmentation
            im, se = load_data.load_one_image(EXAMPLE_IMAGE_ID)
            one_image_pred = sess.run(image_eval, feed_dict={images_ph: im, labels_ph: se, dropout_keep_prob_ph: 1.0})
            load_data.save_image(EXAMPLE_IMAGE_ID, one_image_pred, gs, SAMPLE_IMAGE_PATH)

def get_logger():
    # create logger
    logger = logging.getLogger('sem_segm')
    logger.setLevel(logging.DEBUG)
    hdlr = logging.FileHandler(LOG_FILE)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    return logger
        
if __name__ == '__main__':
    main()
    