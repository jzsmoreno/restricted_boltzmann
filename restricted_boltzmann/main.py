import os
from typing import List, Union

import matplotlib.pyplot as plt
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
        if not isinstance(data, tf.Tensor):
            data = tf.convert_to_tensor(data, dtype=tf.float32)

        # Split the dataset into train and test sets
        train_data, test_data = train_test_split(data.numpy(), test_size=test_size, random_state=42)
        train_data = tf.convert_to_tensor(train_data, dtype=tf.float32)
        test_data = tf.convert_to_tensor(test_data, dtype=tf.float32)

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

        best_test_error = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Shuffle the training data
            shuffled_indices = tf.random.shuffle(tf.range(tf.shape(train_data)[0]))
            train_data_shuffled = tf.gather(train_data, shuffled_indices)

            for start, end in zip(
                range(0, len(train_data), batch_size),
                range(batch_size, len(train_data) + 1, batch_size),
            ):
                batch = train_data_shuffled[start:end]
                w_grad, vb_grad, hb_grad = self._contrastive_divergence(batch)

                # Update weights and biases with the current learning rate
                self.W.assign_add(alpha * w_grad)
                self.vb.assign_add(alpha * vb_grad)
                self.hb.assign_add(alpha * hb_grad)

            # Compute reconstruction error for training data
            v_reconstructed_train = self._sample_v_given_h(self._sample_h_given_v(train_data))
            train_error = tf.reduce_mean(tf.square(train_data - v_reconstructed_train)).numpy()
            train_errors.append(train_error)

            # Compute reconstruction error for test data
            v_reconstructed_test = self._sample_v_given_h(self._sample_h_given_v(test_data))
            test_error = tf.reduce_mean(tf.square(test_data - v_reconstructed_test)).numpy()
            test_errors.append(test_error)

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_error:.4f}, Test Loss: {test_error:.4f}"
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
