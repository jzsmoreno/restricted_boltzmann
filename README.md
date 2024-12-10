![GitHub last commit](https://img.shields.io/github/last-commit/jzsmoreno/restricted_boltzmann?style=for-the-badge)
![GitHub repo size](https://img.shields.io/github/repo-size/jzsmoreno/restricted_boltzmann?style=for-the-badge)
![License](https://img.shields.io/github/license/jzsmoreno/restricted_boltzmann?style=for-the-badge)
[![CI - Test](https://github.com/jzsmoreno/restricted_boltzmann/actions/workflows/code-quality.yml/badge.svg)](https://github.com/jzsmoreno/restricted_boltzmann/actions/workflows/code-quality.yml)


# restricted_boltzmann
Package for the implementation of restricted Boltzmann machines 

## Installation

This package can be easily installed with pip:
```bash
pip install git+https://github.com/jzsmoreno/restricted_boltzmann
```

## Features

- **Training**: Implements Contrastive Divergence (CD) and other training algorithms for RBMs.
- **Inference**: Support for both Gibbs sampling and deterministic inference methods.
- **Customization**: Allows the configuration of various hyperparameters such as the number of hidden units, learning rate, and batch size.
- **Compatibility**: Can be easily integrated with popular deep learning frameworks like TensorFlow or PyTorch for more advanced use cases.

## Applications

RBMs are widely used in various real-world applications, including:

- **Pattern Recognition**: Feature extraction in pattern recognition problems, such as handwriting recognition or pattern classification.
- **Recommendation Systems**: Collaborative filtering for recommending products, movies, or other items based on user behavior (e.g., movie or book recommendations).
- **Radar Target Recognition**: Used in radar systems to detect low signal-to-noise ratio (SNR) targets in environments with high noise levels.

## Key Features of RBMs

- **Unsupervised Learning**: RBMs learn from input data without requiring labeled responses.
- **Recurrent and Symmetric Structure**: The network structure is symmetric, with the same types of connections between visible and hidden layers.
- **No Intra-Layer Connections**: There are no connections within the visible layer or within the hidden layer, making the model computationally efficient.
- **Energy-Based Model**: RBMs associate high probability with low-energy configurations.

## Contributing

If you'd like to contribute to the development of this package, feel free to fork the repository and submit a pull request. Please make sure your contributions adhere to the coding style and include relevant tests.

## License

This package is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.