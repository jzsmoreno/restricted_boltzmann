from typing import List

import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import clear_output


class RestrictedBoltzmann:
    """A class that implements a Restricted Boltzmann Machine"""

    def __init__(self):
        self.vb = None
        self.hb = None
        self.W = None
        self.hiddenunits = None
        self.visibleunits = None

    def _h0(self, v0: List[List]):
        return tf.nn.sigmoid(tf.matmul(v0, self.W) + self.hb)

    def h0(self, v0: List[List]):
        return tf.nn.relu(tf.sign(self._h0(v0) - tf.random.uniform(tf.shape(self._h0(v0)))))

    def _v1(self, v0: List[List]):
        return tf.nn.sigmoid(tf.matmul(self.h0(v0), tf.transpose(self.W)) + self.vb)

    def v1(self, v0: List[List]):
        return tf.nn.relu(tf.sign(self._v1(v0) - tf.random.uniform(tf.shape(self._v1(v0)))))

    def _h1(self, v0: List[List]):
        return tf.sigmoid(tf.matmul(self.v1(v0), self.W) + self.hb)

    def h1(self, v0: List[List]):
        return tf.nn.relu(tf.sign(self._h1(v0) - tf.random.uniform(tf.shape(self._h1(v0)))))

    def hh0(self, v0: List[List]):
        return tf.nn.sigmoid(tf.matmul(v0, self.W) + self.hb)

    def vv1(self, v0: List[List]):
        return tf.nn.sigmoid(tf.matmul(self.hh0(v0), tf.transpose(self.W)) + self.vb)

    def train(
        self,
        v0: List[List],
        hiddenunits: int,
        visibleunits: int,
        alpha: float = 1.0,
        epochs: int = 25,
        batchsize: int = 100,
        plot: bool = True,
    ):
        self.hiddenunits = hiddenunits
        self.visibleunits = visibleunits
        self.vb = tf.constant(0.1, shape=[self.visibleunits])
        self.hb = tf.constant(0.1, shape=[self.hiddenunits])
        self.W = tf.Variable(
            tf.random.truncated_normal([self.visibleunits, self.hiddenunits], stddev=0.1)
        )

        cur_w = tf.Variable(tf.zeros([self.visibleunits, self.hiddenunits], tf.float32))
        cur_vb = tf.Variable(tf.zeros([self.visibleunits], tf.float32))
        cur_hb = tf.Variable(tf.zeros([self.hiddenunits], tf.float32))
        prv_w = tf.Variable(tf.zeros([self.visibleunits, self.hiddenunits], tf.float32))
        prv_vb = tf.Variable(tf.zeros([self.visibleunits], tf.float32))
        prv_hb = tf.Variable(tf.zeros([self.hiddenunits], tf.float32))

        def w_pos_grad(v0):
            return tf.matmul(tf.transpose(v0), self.h0(v0))

        def w_neg_grad(v0):
            return tf.matmul(tf.transpose(self.v1(v0)), self.h1(v0))

        def cd(v0):
            return (w_pos_grad(v0) - w_neg_grad(v0)) / tf.cast(tf.shape(v0)[0], "float32")

        def update_w(v0):
            return self.W + alpha * cd(v0)

        def update_vb(v0):
            return self.vb + alpha * tf.reduce_mean(v0 - self.v1(v0), 0)

        def update_hb(v0):
            return self.hb + alpha * tf.reduce_mean(self.h0(v0) - self.h1(v0), 0)

        def err(v0):
            return v0 - self.v1(v0)

        def err_sum(v0):
            return tf.reduce_mean(err(v0) * err(v0))

        errors = []
        for i in range(epochs):
            for start, end in zip(
                range(0, len(v0), batchsize), range(batchsize, len(v0), batchsize)
            ):
                batch = v0[start:end]
                batch = tf.cast(batch, "float32")
                cur_w = update_w(batch)
                cur_vb = update_vb(batch)
                cur_hb = update_hb(batch)
                prv_w = cur_w
                prv_vb = cur_vb
                prv_hb = cur_hb
                self.W = prv_w
                self.vb = prv_vb
                self.hb = prv_hb

            errors.append(err_sum(batch))

            if plot:
                clear_output(wait=True)
                plt.plot(errors)
                plt.ylabel("Error")
                plt.xlabel("Epoch")
                plt.show()

    def predict(self, v0):
        return self.vv1(v0).numpy()
