# GRYDRA

GRYDRA is a Ruby gem designed for building, training, and utilizing neural networks. It provides a flexible framework for creating multi-subnet architectures, implementing various activation functions, optimization techniques, and preprocessing tools, suitable for numerical, categorical (hash), and text-based data tasks.

## Features

*   **Neural Network Architectures:** Build standard feedforward networks, multi-subnet ensembles (`MainNetwork`), and simplified interfaces (`EasyNetwork`).
*   **Multiple Activation Functions:** Includes Tanh, Sigmoid, ReLU, Leaky ReLU, Swish, GELU, and more.
*   **Advanced Optimizers:** Supports the Adam optimizer for efficient training.
*   **Regularization Techniques:** Includes L1, L2 regularization, and Dropout.
*   **Data Preprocessing:** Offers Min-Max and Z-Score normalization. Includes utilities for handling categorical (hash) and text data (vocabulary creation, vectorization, TF-IDF).
*   **Training Features:** Supports mini-batch training, early stopping, learning rate decay, and customizable parameters.
*   **Evaluation Metrics:** Provides MSE, MAE, Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and AUC-ROC.
*   **Cross-Validation:** Includes k-fold cross-validation for robust model evaluation.
*   **Model Persistence:** Save and load trained models and vocabularies using Ruby's `Marshal`.
*   **Analysis Tools:** Gradient analysis and ASCII visualization of network architecture.
*   **Hyperparameter Search:** Basic grid search functionality.

## Installation

1.  Ensure you have Ruby installed.
2.  Save the provided code as a `.rb` file (e.g., `grydra.rb`) or create a Ruby gem.
3.  Require the file or gem in your project: `require 'grydra'` (or the path to your file).

## Usage

### Basic Example: Training with Numerical Data

```ruby
require 'grydra'

# Example data: [height (cm), age (years)] -> [weight (kg)]
data_input = [[170, 25], [160, 30], [180, 22]]
data_output = [[65], [60], [75]]

# Define subnet structures (hidden layers)
structures = [[4, 1], [3, 1]]

# Create the network using the easy interface
network = GRYDRA::EasyNetwork.new(print_epochs = true)

# Train the network
network.train_numerical(
  data_input,
  data_output,
  structures,
  learning_rate = 0.05,
  epochs = 15000,
  normalization = :max # Options: :max, :zscore
)

# Make a prediction for a new individual
new_data = [[172, 26]]
predictions = network.predict_numerical(new_data, :max) # Use same normalization

puts "Predicted weight: #{predictions[0][0].round(2)} kg"
```

### Basic Example: Training with Hash Data

```ruby
require 'grydra'

# Example data: Categorical inputs mapped to a numerical label
data_hash = [
  { height: 170, is_new: false, label: 0 },
  { height: 160, is_new: true,  label: 1 },
  { height: 180, is_new: false, label: 0 },
]

input_keys = [:height, :is_new]
label_key = :label
structures = [[3, 1]]

network = GRYDRA::EasyNetwork.new(print_epochs = true)

network.train_hashes(
  data_hash,
  input_keys,
  label_key,
  structures,
  learning_rate = 0.05,
  epochs = 10000,
  normalization = :max
)

# Predict for new data
new_hashes = [{ height: 175, is_new: true }]
predictions = network.predict_hashes(new_hashes, input_keys, :max)

puts "Prediction: #{predictions[0][0].round(3)}"
```

### Key Classes

*   `GRYDRA::EasyNetwork`: A high-level interface for easier training and prediction on numerical, hash, and text data.
*   `GRYDRA::MainNetwork`: A class for managing and training multiple sub-networks.
*   `GRYDRA::NeuralNetwork`: Represents a single neural network (used internally by `MainNetwork`).
*   `GRYDRA::Neuron`: Represents a single neuron within a layer.
*   `GRYDRA::DenseLayer`: A standard fully connected layer.
*   `GRYDRA::AdamOptimizer`: An implementation of the Adam optimizer.

### Helper Functions

*   `GRYDRA.save_model(model, name, path, vocabulary)`: Saves a trained model.
*   `GRYDRA.load_model(name, path)`: Loads a saved model.
*   `GRYDRA.save_vocabulary(vocabulary, name, path)`: Saves a vocabulary.
*   `GRYDRA.load_vocabulary(name, path)`: Loads a vocabulary.
*   `GRYDRA.describe_method(class_name, method_name)`: Provides information and examples for specific methods.
*   `GRYDRA.list_methods_available()`: Lists all documented public methods.
*   `GRYDRA.generate_example(num, filename, ext, path)`: Generates example scripts demonstrating usage.

## Examples

The `GRYDRA.generate_example` method can create various example scripts (numbered 1-12) showcasing different features like advanced training, text processing, classification metrics, and cross-validation. Run `GRYDRA.generate_example(1)` to start.

## License

[Show licence](LICENCE)