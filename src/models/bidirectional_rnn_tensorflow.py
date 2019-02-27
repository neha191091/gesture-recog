import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn, dynamic_rnn


class BidirectionalRNNTf:
    def __init__(self, largest_seq_len, max_iter, num_classes, num_features, direction='both'):

        self.num_features = num_features
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.largest_seq_len = largest_seq_len

        # tf Graph input

        self.batch_size = tf.placeholder(tf.int64)

        self.X = tf.placeholder(tf.float32, [None, largest_seq_len, num_features])
        self.t = tf.placeholder(tf.int32, [None])
        self.sequence_len = tf.placeholder(tf.int32, [None])

        dataset = tf.data.Dataset.from_tensor_slices((self.X, self.t, self.sequence_len))
        dataset = dataset.batch(self.batch_size)
        self.iterator = dataset.make_initializable_iterator()

        X, t, sequence_len = self.iterator.get_next()
        t = tf.cast(t, tf.int32)
        sequence_len = tf.cast(sequence_len, tf.int32)

        cell_size = 5

        # Forward direction cell
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(cell_size)

        # Backward direction cell
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(cell_size)

        outputs, _ = bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, X,
                                                           sequence_length=sequence_len,
                                                           dtype=tf.float32)
        outputs_fw, outputs_bw = outputs

        tf_batch_sz = tf.shape(outputs_fw)[0]
        outputs_fw = tf.reshape(outputs_fw, [-1, cell_size])
        indices = tf.range(0, tf_batch_sz) * largest_seq_len + (sequence_len - 1)
        outputs_fw = tf.gather(outputs_fw, indices)

        tf_batch_sz = tf.shape(outputs_bw)[0]
        outputs_bw = tf.reshape(outputs_bw, [-1, cell_size])
        indices = tf.range(0, tf_batch_sz) * largest_seq_len + (sequence_len - 1)
        outputs_bw = tf.gather(outputs_bw, indices)

        if direction == 'both':
            outputs = tf.concat([outputs_bw, outputs_fw], axis=-1)
        elif direction == 'forward':
            outputs = outputs_fw
        else:
            outputs = outputs_bw
        print(outputs)

        outputs = tf.layers.dense(outputs, num_classes)

        self.loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=t))

        # Evaluate model (with test logits, for dropout to be disabled)

        self.outputs = tf.argmax(outputs, 1, output_type=tf.int32)

        correct_pred = tf.equal(self.outputs, t)

        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def fit(self, X, t, seq_len, X_val, t_val, seq_len_val, epochs, batch_size, lr):

        X = X.reshape([X.shape[0], -1, self.num_features])
        X_val = X_val.reshape([X_val.shape[0], -1, self.num_features])

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(self.loss_op)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch_no in range(epochs):
                train_loss, train_accuracy = 0, 0
                val_loss, val_accuracy = 0, 0
                train_steps = 0
                val_steps = 0

                # Initialize iterator with training data
                sess.run(self.iterator.initializer, feed_dict={self.X: X, self.t: t, self.sequence_len: seq_len, self.batch_size: batch_size})
                try:
                    with tqdm(total=len(t)) as pbar:
                        while True:
                            _, loss, acc = sess.run([train_op, self.loss_op, self.accuracy])
                            train_loss += loss
                            train_accuracy += acc
                            train_steps += 1
                            pbar.update(batch_size)
                except tf.errors.OutOfRangeError:
                    pass

                # Initialize iterator with validation data
                sess.run(self.iterator.initializer, feed_dict={self.X: X_val, self.t: t_val, self.sequence_len: seq_len_val, self.batch_size: batch_size})
                try:
                    while True:
                        loss, acc = sess.run([self.loss_op, self.accuracy])
                        val_loss += loss
                        val_accuracy += acc
                        val_steps += 1
                except tf.errors.OutOfRangeError:
                    pass

                print('\nEpoch No: {}'.format(epoch_no + 1))
                print('Train accuracy = {:.4f}, loss = {:.4f}'.format(train_accuracy / train_steps,
                                                                      train_loss / train_steps))
                print('Val accuracy = {:.4f}, loss = {:.4f}'.format(val_accuracy / val_steps,
                                                                    val_loss / val_steps))

                if not os.path.exists('checkpoints'):
                    os.mkdir('checkpoints')
                save_path = saver.save(sess, "checkpoints/model.ckpt")
                print("Model saved in path: %s" % save_path)

    def score(self, X, t, seq_len, batch_size=20):

        X = X.reshape([X.shape[0], -1, self.num_features])

        saver = tf.train.Saver()

        with tf.Session() as sess:

            saver.restore(sess, "checkpoints/model.ckpt")
            print("Model restored.")

            # Initialize iterator with training data
            total_loss, total_acc = 0, 0
            total_steps = 0

            # Initialize iterator with training data
            sess.run(self.iterator.initializer,
                     feed_dict={self.X: X, self.t: t, self.sequence_len: seq_len, self.batch_size: batch_size})
            try:
                while True:
                    loss, acc = sess.run([self.loss_op, self.accuracy])
                    total_loss += loss
                    total_acc += acc
                    total_steps += 1
            except tf.errors.OutOfRangeError:
                pass

            #print('Accuracy = {:.4f}, loss = {:.4f}'.format(total_acc/total_steps, total_loss/total_steps))
        return total_loss/total_steps, total_acc/total_steps

    def predict(self, X, t, seq_len):

        X = X.reshape([X.shape[0], -1, self.num_features])

        saver = tf.train.Saver()

        preds=[]

        # Start training
        with tf.Session() as sess:

            saver.restore(sess, "checkpoints/model.ckpt")
            print("Model restored.")

            # Initialize iterator with training data
            sess.run(self.iterator.initializer,
                     feed_dict={self.X: X, self.t: t, self.sequence_len: seq_len, self.batch_size: X.shape[0]})
            try:
                while True:
                    preds = sess.run([self.outputs])
            except tf.errors.OutOfRangeError:
                pass

        return preds[0]
