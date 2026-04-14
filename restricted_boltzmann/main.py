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
        """Computes the sigmoid function.

        Parameters
        ----------
        x : `tf.Tensor` or `np.ndarray`
            Input tensor or array to apply the sigmoid function.

        Returns
        -------
        output : `tf.Tensor`
            The sigmoid of the input values.
        """
        return tf.nn.sigmoid(x)

    def _sample_h_given_v(self, v):
        """Samples hidden units given visible units.

        Parameters
        ----------
        v : `tf.Tensor` or `np.ndarray`
            Visible unit activations.

        Returns
        -------
        h_samples : `tf.Tensor`
            Sampled hidden unit activations (binary).
        """
        probabilities = self._sigmoid(tf.matmul(v, self.W) + self.hb)
        return tf.where(probabilities > tf.random.uniform(tf.shape(probabilities)), 1.0, 0.0)

    def _sample_v_given_h(self, h):
        """Samples visible units given hidden units.

        Parameters
        ----------
        h : `tf.Tensor` or `np.ndarray`
            Hidden unit activations.

        Returns
        -------
        v_samples : `tf.Tensor`
            Sampled visible unit activations (binary).
        """
        probabilities = self._sigmoid(tf.matmul(h, tf.transpose(self.W)) + self.vb)
        return tf.where(probabilities > tf.random.uniform(tf.shape(probabilities)), 1.0, 0.0)

    def _compute_free_energy(self, v):
        """Computes the free energy of the visible units.

        Parameters
        ----------
        v : `tf.Tensor` or `np.ndarray`
            Visible unit activations.

        Returns
        -------
        free_energy : `tf.Tensor`
            The computed free energy values for each sample.
        """
        if not isinstance(v, tf.Tensor):
            v = tf.convert_to_tensor(v, dtype=tf.float32)
        vb_term = tf.reduce_sum(v * self.vb, axis=1)
        wx_b_term = tf.matmul(v, self.W) + self.hb
        hidden_term = tf.reduce_sum(tf.math.log(1 + tf.exp(wx_b_term)), axis=1)
        return -vb_term - hidden_term

    def _contrastive_divergence(self, v0):
        """Performs contrastive divergence learning step.

        Parameters
        ----------
        v0 : `tf.Tensor` or `np.ndarray`
            Initial visible unit activations (training data batch).

        Returns
        -------
        w_grad : `tf.Tensor`
            Weight gradient for update.
        vb_grad : `tf.Tensor`
            Visible bias gradient for update.
        hb_grad : `tf.Tensor`
            Hidden bias gradient for update.
        """
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
        """Computes the reconstruction accuracy between original and reconstructed data.

        Parameters
        ----------
        original_data : `tf.Tensor` or `np.ndarray`
            Original input data.
        reconstructed_data : `tf.Tensor` or `np.ndarray`
            Reconstructed output data from the model.

        Returns
        -------
        accuracy : `float`
            The reconstruction accuracy as a fraction of correct predictions.
        """
        # Assuming binary data
        return tf.reduce_mean(
            tf.cast(tf.equal(original_data, reconstructed_data), tf.float32)
        ).numpy()

    def get_hidden_activations(self, data: Union[List[List[float]], tf.Tensor]) -> np.ndarray:
        """Extract hidden layer activations for the given input data.

        Parameters
        ----------
        data : `list` or `tf.Tensor`
            Input data as a list of lists or TensorFlow tensor.

        Returns
        -------
        activations : `np.ndarray`
            Hidden layer activation values for each sample.
        """
        if not isinstance(data, tf.Tensor):
            data = tf.convert_to_tensor(data, dtype=tf.float32)
        return self._sigmoid(tf.matmul(data, self.W) + self.hb).numpy()

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
        """Trains the Restricted Boltzmann Machine on the provided data.

        Parameters
        ----------
        data : `list` or `tf.Tensor`
            Training data as a list of lists or TensorFlow tensor.
        hidden_units : `int`
            Number of hidden units in the RBM.
        visible_units : `int`
            Number of visible units (matching input dimension).
        alpha : `float`, optional
            Learning rate for weight updates (default: 1.0).
        epochs : `int`, optional
            Number of training epochs (default: 25).
        batch_size : `int`, optional
            Size of batches for training (default: 100).
        plot : `bool`, optional
            Whether to plot reconstruction errors during training (default: True).
        verbose : `bool`, optional
            Whether to print training progress (default: False).
        test_size : `float`, optional
            Fraction of data to use for testing (default: 0.2).
        early_stopping_patience : `int`, optional
            Number of epochs without improvement before stopping (default: 5).
        decay_rate : `float`, optional
            Learning rate decay factor when early stopping triggers (default: 0.95).

        Returns
        -------
        self : `RestrictedBoltzmann`
            The trained RBM model instance.
        """
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

        # Initialize weights and biases if self.W, self.vb and self.hb are None
        if self.W is None and self.vb is None and self.hb is None:
            self.W = tf.Variable(
                tf.random.truncated_normal([self.visible_units, self.hidden_units], stddev=0.1)
            )
            self.vb = tf.Variable(tf.zeros([self.visible_units]))
            self.hb = tf.Variable(tf.zeros([self.hidden_units]))
            if verbose:
                print("Weights, visible biases and hidden biases were initialized.")
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
        """Save the model's weights and biases to a checkpoint directory.

        Parameters
        ----------
        checkpoint_dir : `str`
            Directory path where the model checkpoint will be saved.
        """
        checkpoint = tf.train.Checkpoint(W=self.W, vb=self.vb, hb=self.hb)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint.save(file_prefix=checkpoint_prefix)

    def load_model(self, checkpoint_dir: str, hidden_units: int, visible_units: int):
        """Load the model's weights and biases from a checkpoint directory.

        Parameters
        ----------
        checkpoint_dir : `str`
            Directory path containing the saved model checkpoint.
        hidden_units : `int`
            Number of hidden units in the RBM (must match original training).
        visible_units : `int`
            Number of visible units (must match original training).

        Returns
        -------
        self : `RestrictedBoltzmann`
            The loaded RBM model instance with restored weights and biases.
        """
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
        """Predicts visible unit activations for given input data.

        Parameters
        ----------
        data : `list` or `tf.Tensor`
            Input data as a list of lists or TensorFlow tensor.

        Returns
        -------
        predictions : `list`
            Predicted visible unit activations (binary) for each sample.
        """
        if not isinstance(data, tf.Tensor):
            data = tf.convert_to_tensor(data, dtype=tf.float32)
        return self._sample_v_given_h(self._sample_h_given_v(data)).numpy()

    def predict_proba(self, data: Union[List[List[float]], tf.Tensor]) -> List[List[float]]:
        """Predicts probabilities of visible unit activations for given input data.

        Parameters
        ----------
        data : `list` or `tf.Tensor`
            Input data as a list of lists or TensorFlow tensor.

        Returns
        -------
        probabilities : `list`
            Predicted probabilities (between 0 and 1) for each visible unit.
        """
        if not isinstance(data, tf.Tensor):
            data = tf.convert_to_tensor(data, dtype=tf.float32)
        return self._sigmoid(
            tf.matmul(self._sigmoid(tf.matmul(data, self.W) + self.hb), tf.transpose(self.W))
            + self.vb
        ).numpy()

    def summary(self):
        """Prints a detailed summary of the model architecture and parameters.

        Displays layer types, shapes, parameter counts, and total model size.
        Prints "Model has not been initialized yet." if weights are not set.
        """
        if self.W is None or self.vb is None or self.hb is None:
            print("Model has not been initialized yet.")
            return

        # Calculate the number of parameters for each component
        num_weights = np.prod(self.W.shape)
        num_visible_biases = self.vb.numpy().size
        num_hidden_biases = self.hb.numpy().size
        total_params = num_weights + num_visible_biases + num_hidden_biases

        # Model details
        visible_shape = tuple(self.vb.shape)
        hidden_shape = tuple(self.hb.shape)

        # Print the summary in a more detailed and formatted manner
        print("=" * 50)
        print(f"{'Model Summary':^50}")
        print("=" * 50)
        print(f"{'Layer':<20} {'Type':<15} {'Shape':<25} {'Parameters':<25}")
        print("-" * 75)

        # Visible layer
        visible_shape_str = str(visible_shape).replace(",", ", ")
        print(
            f"{'Visible Layer':<20} {'Bias':<15} {visible_shape_str:<25} {num_visible_biases:<25}"
        )

        # Weights between layers
        weight_shape = tuple(self.W.shape)
        weight_shape_str = str(weight_shape).replace(",", ", ")
        print(f"{'(Vis -> Hidden)':<20} {'Weight':<15} {weight_shape_str:<25} {num_weights:<25}")

        # Hidden layer
        hidden_shape_str = str(hidden_shape).replace(",", ", ")
        print(f"{'Hidden Layer':<20} {'Bias':<15} {hidden_shape_str:<25} {num_hidden_biases:<25}")

        print("-" * 75)
        print(f"{'Total Layers':<40} {2:<25}")
        print(f"{'Total Parameters':<40} {total_params:<25}")
        print(f"{'Model Size (Approx)':<40} {total_params * 4 / (1024**2):.2f} MB")
        print("=" * 50)

    def plot_distributions(self, title="Distributions of Weights and Biases"):
        """Plots histograms of weights and biases distributions.

        Parameters
        ----------
        title : `str`, optional
            Title for the plot (default: "Distributions of Weights and Biases").

        Returns
        -------
        fig, axes : `matplotlib.figure.Figure`, `list`
            The figure object and axes containing the distribution plots.
        """
        plt.figure(figsize=(12, 4))

        # Plot weights distribution
        plt.subplot(1, 3, 1)
        plt.hist(self.W.numpy().flatten(), bins=50, alpha=0.75, color="blue")
        plt.title("Weights Distribution")

        # Plot visible biases distribution
        plt.subplot(1, 3, 2)
        plt.hist(self.vb.numpy().flatten(), bins=50, alpha=0.75, color="green")
        plt.title("Visible Biases Distribution")

        # Plot hidden biases distribution
        plt.subplot(1, 3, 3)
        plt.hist(self.hb.numpy().flatten(), bins=50, alpha=0.75, color="red")
        plt.title("Hidden Biases Distribution")

        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        plt.show()

    def summarize_statistics(self):
        """Prints summary statistics for weights and biases.

        Displays mean, standard deviation, and sparsity metrics for each component.
        Prints a formatted table with statistical summaries of the model parameters.
        """
        print("=" * 50)
        print(f"{'Summary Statistics':^50}")
        print("=" * 50)
        # Weights statistics
        weights = self.W.numpy()
        print("\nWeights:")
        print("-" * 75)
        print(
            f"Mean: {np.mean(weights):>10.4f}, Std: {np.std(weights):>10.4f}, Sparsity: {np.sum(weights == 0) / len(weights.flatten()):>10.4f}"
        )

        # Visible biases statistics
        visible_biases = self.vb.numpy()
        print("\nVisible Biases:")
        print("-" * 75)
        print(f"Mean: {np.mean(visible_biases):>10.4f}, Std: {np.std(visible_biases):>10.4f}")

        # Hidden biases statistics
        hidden_biases = self.hb.numpy()
        print("\nHidden Biases:")
        print("-" * 75)
        print(f"Mean: {np.mean(hidden_biases):>10.4f}, Std: {np.std(hidden_biases):>10.4f}")
        print("=" * 50)
