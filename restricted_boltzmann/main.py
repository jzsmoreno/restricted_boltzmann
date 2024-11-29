import os
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import clear_output
from sklearn.model_selection import train_test_split


class RestrictedBoltzmann:
    """A class that implements a Restricted Boltzmann Machine"""

    def __init__(self):
        self.vb = None
        self.hb = None
        self.W = None
        self.hidden_units = None
        self.visible_units = None

    def _sigmoid(self, x):
        return tf.nn.sigmoid(x)

    def _sample_h_given_v(self, v):
        probabilities = self._sigmoid(tf.matmul(v, self.W) + self.hb)
        return tf.where(probabilities > tf.random.uniform(tf.shape(probabilities)), 1.0, 0.0)

    def _sample_v_given_h(self, h):
        probabilities = self._sigmoid(tf.matmul(h, tf.transpose(self.W)) + self.vb)
        return tf.where(probabilities > tf.random.uniform(tf.shape(probabilities)), 1.0, 0.0)

    def _compute_free_energy(self, v):
        if not isinstance(v, tf.Tensor):
            v = tf.convert_to_tensor(v, dtype=tf.float32)
        vb_term = tf.reduce_sum(v * self.vb, axis=1)
        wx_b_term = tf.matmul(v, self.W) + self.hb
        hidden_term = tf.reduce_sum(tf.math.log(1 + tf.exp(wx_b_term)), axis=1)
        return -vb_term - hidden_term

    def _contrastive_divergence(self, v0):
        h0 = self._sample_h_given_v(v0)
        vk = v0  # Start with the original data
        hk = None
        for _ in range(1):  # Typically k=1 is used
            hk = self._sample_h_given_v(vk)
            vk = self._sample_v_given_h(hk)

        w_positive_grad = tf.matmul(tf.transpose(v0), h0) / tf.cast(tf.shape(v0)[0], tf.float32)
        w_negative_grad = tf.matmul(tf.transpose(vk), hk) / tf.cast(tf.shape(vk)[0], tf.float32)
        return (
            w_positive_grad - w_negative_grad,
            tf.reduce_mean(v0 - vk, axis=0),
            tf.reduce_mean(h0 - hk, axis=0),
        )

    def _compute_reconstruction_accuracy(self, original_data, reconstructed_data):
        # Assuming binary data
        return tf.reduce_mean(
            tf.cast(tf.equal(original_data, reconstructed_data), tf.float32)
        ).numpy()

    def train(
        self,
        data: Union[List[List[float]], tf.Tensor],
        hidden_units: int,
        visible_units: int,
        alpha: float = 1.0,
        epochs: int = 25,
        batch_size: int = 100,
        plot: bool = True,
        verbose: bool = False,
        test_size: float = 0.2,
        early_stopping_patience: int = 5,
        decay_rate: float = 0.95,  # Exponential decay rate
    ):
        if isinstance(data, tf.Tensor):
            data = data.numpy()
        elif not isinstance(data, np.ndarray):
            data = np.array(data)

        data = data.astype(np.float32)

        # Split the dataset into train and test sets
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
        train_dataset = (
            tf.data.Dataset.from_tensor_slices(train_data)
            .shuffle(buffer_size=len(train_data))
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        test_dataset = tf.data.Dataset.from_tensor_slices(test_data).batch(batch_size)

        self.hidden_units = hidden_units
        self.visible_units = visible_units

        # Initialize weights and biases
        self.W = tf.Variable(
            tf.random.truncated_normal([self.visible_units, self.hidden_units], stddev=0.1)
        )
        self.vb = tf.Variable(tf.zeros([self.visible_units]))
        self.hb = tf.Variable(tf.zeros([self.hidden_units]))

        train_errors = []
        test_errors = []
        train_accuracies = []
        test_accuracies = []

        best_test_error = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            for batch in train_dataset:
                w_grad, vb_grad, hb_grad = self._contrastive_divergence(batch)

                # Update weights and biases with the current learning rate
                self.W.assign_add(alpha * w_grad)
                self.vb.assign_add(alpha * vb_grad)
                self.hb.assign_add(alpha * hb_grad)

            # Compute reconstruction error for training data
            train_error = 0.0
            train_accuracy = 0.0
            num_batches_train = 0

            for batch in train_dataset:
                v_reconstructed_batch = self._sample_v_given_h(self._sample_h_given_v(batch))
                train_error += tf.reduce_mean(tf.square(batch - v_reconstructed_batch)).numpy()
                train_accuracy += self._compute_reconstruction_accuracy(
                    batch, v_reconstructed_batch
                )
                num_batches_train += 1

            train_error /= num_batches_train
            train_accuracy /= num_batches_train
            train_errors.append(train_error)
            train_accuracies.append(train_accuracy)

            # Compute reconstruction error for test data
            test_error = 0.0
            test_accuracy = 0.0
            num_batches_test = 0

            for batch in test_dataset:
                v_reconstructed_batch = self._sample_v_given_h(self._sample_h_given_v(batch))
                test_error += tf.reduce_mean(tf.square(batch - v_reconstructed_batch)).numpy()
                test_accuracy += self._compute_reconstruction_accuracy(batch, v_reconstructed_batch)
                num_batches_test += 1

            test_error /= num_batches_test
            test_accuracy /= num_batches_test
            test_errors.append(test_error)
            test_accuracies.append(test_accuracy)

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_error:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_error:.4f}, Test Accuracy: {test_accuracy:.4f}"
                )

            if plot and (epoch % 5 == 0 or epoch == epochs - 1):
                clear_output(wait=True)
                plt.plot(train_errors, label="Train Reconstruction Error", color="blue")
                plt.plot(test_errors, label="Test Reconstruction Error", color="red")
                plt.ylabel("Error")
                plt.xlabel("Epoch")
                plt.legend()
                plt.show()

            # Early stopping logic
            if test_error < best_test_error:
                best_test_error = test_error
                patience_counter = 0
                # Decay the learning rate for the next epoch
                alpha *= decay_rate
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    def save_model(self, checkpoint_dir: str):
        """Save the model's weights and biases to a checkpoint directory."""
        checkpoint = tf.train.Checkpoint(W=self.W, vb=self.vb, hb=self.hb)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint.save(file_prefix=checkpoint_prefix)

    def load_model(self, checkpoint_dir: str, hidden_units: int, visible_units: int):
        """Load the model's weights and biases from a checkpoint directory."""
        self.hidden_units = hidden_units
        self.visible_units = visible_units
        checkpoint = tf.train.Checkpoint(
            W=tf.Variable(tf.zeros([self.visible_units, self.hidden_units])),
            vb=tf.Variable(tf.zeros([self.visible_units])),
            hb=tf.Variable(tf.zeros([self.hidden_units])),
        )
        status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        status.assert_consumed()
        self.W = checkpoint.W
        self.vb = checkpoint.vb
        self.hb = checkpoint.hb

    def predict(self, data: Union[List[List[float]], tf.Tensor]) -> List[List[float]]:
        if not isinstance(data, tf.Tensor):
            data = tf.convert_to_tensor(data, dtype=tf.float32)
        return self._sample_v_given_h(self._sample_h_given_v(data)).numpy()

    def predict_proba(self, data: Union[List[List[float]], tf.Tensor]) -> List[List[float]]:
        if not isinstance(data, tf.Tensor):
            data = tf.convert_to_tensor(data, dtype=tf.float32)
        return self._sigmoid(
            tf.matmul(self._sigmoid(tf.matmul(data, self.W) + self.hb), tf.transpose(self.W))
            + self.vb
        ).numpy()
