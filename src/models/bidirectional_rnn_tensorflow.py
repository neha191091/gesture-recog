import tensorflow as tf
import numpy as np

from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn


class BidirectionalRNNTf:
    def __init__(self, dataset, max_iter, num_classes, num_features):

        self.num_features = num_features
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.dataset = dataset

        # tf Graph input
        self.X = tf.placeholder(tf.float32, [None, dataset.largest_seq_len, num_features])
        self.t = tf.placeholder(tf.int32, [None])
        self.sequence_len = tf.placeholder(tf.int32, [None])


        # Forward direction cell
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(5, forget_bias=1.0)

        # Backward direction cell
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(5, forget_bias=1.0)

        outputs, _ = bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.X,
                                                           sequence_length=self.sequence_len,
                                                           dtype=tf.float32)
        outputs_fw, outputs_bw = outputs

        outputs_fw = tf.transpose(outputs_fw, perm=(1, 0, 2))
        outputs_bw = tf.transpose(outputs_bw, perm=(1, 0, 2))

        outputs = tf.concat([outputs_bw[-1], outputs_fw[-1]], axis=-1)

        outputs = tf.layers.dense(outputs, num_classes)

        print(outputs)

        self.loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=self.t))

        # Evaluate model (with test logits, for dropout to be disabled)

        self.outputs = tf.argmax(outputs, 1, output_type=tf.int32)

        correct_pred = tf.equal(self.outputs, self.t)

        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def fit(self, normalize=False, learning_rate=5e-1):

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(self.loss_op)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start training
        with tf.Session() as sess:

            # Run the initializer
            sess.run(init)

            for step in range(1, self.max_iter + 1):
                batch_x, batch_t, seq_len = self.dataset.get_padded_batch_flat(split_type='train', normalize=normalize)
                # Reshape data to get 28 seq of 28 elements
                batch_x = batch_x.reshape((batch_x.shape[0], -1, self.num_features))
                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={self.X: batch_x, self.t: batch_t, self.sequence_len: seq_len})
                if step % 10 == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([self.loss_op, self.accuracy], feed_dict={self.X: batch_x,
                                                                                   self.t: batch_t,
                                                                                   self.sequence_len: seq_len})
                    print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))

    def score(self, X, t, seq_len):

        X = X.reshape([X.shape[0], -1, self.num_features])

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start training
        with tf.Session() as sess:
            # Run the initializer
            sess.run(init)

            # Calculate batch loss and accuracy
            loss, acc = sess.run([self.loss_op, self.accuracy], feed_dict={self.X: X,
                                                                           self.t: t,
                                                                           self.sequence_len: seq_len})
            print("Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    def predict(self, X, seq_len):

        X = X.reshape([X.shape[0], -1, self.num_features])

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start training
        with tf.Session() as sess:
            # Run the initializer
            sess.run(init)

            # Calculate batch loss and accuracy
            preds = sess.run([self.outputs], feed_dict={self.X: X,
                                                        self.sequence_len: seq_len})

        return preds
