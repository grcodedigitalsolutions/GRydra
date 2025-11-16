# GRYDRA

![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Ruby](https://img.shields.io/badge/Ruby-3.x-red)
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)
![NeuralNetworks](https://img.shields.io/badge/AI-Neural_Networks-orange)


---

## 📘 Overview

GRYDRA is a Ruby gem for building, training, and deploying neural networks. Supports multi-subnet architectures, advanced activations, optimizers, preprocessing for numerical, hash-based, and text data, evaluation metrics, and model persistence.

---

## 🔧 Features

- Neural network architectures: feedforward, multi-subnet (`MainNetwork`), simplified interface (`EasyNetwork`)
- Activation functions: Tanh, Sigmoid, ReLU, Leaky ReLU, Swish, GELU
- Optimizers: Adam
- Regularization: L1, L2, Dropout
- Preprocessing: Min-Max, Z-Score, hashing utilities, vocabulary creation, TF-IDF
- Training tools: mini-batch, early stopping, learning rate decay
- Evaluation: MSE, MAE, Accuracy, Precision, Recall, F1, Confusion Matrix, AUC-ROC
- Cross-validation: k-fold
- Persistence: save/load models and vocabularies with Ruby `Marshal`
- Analysis: gradient inspection, ASCII visualization
- Hyperparameter tuning: basic grid search

---

## 📥 Installation

### Install via Gem
```bash
gem install grydra
````

### Manual install

```bash
git clone https://github.com/tuusuario/grydra.git
cd grydra
```

### Require in your project

```ruby
require 'grydra'
```

---

## 🚀 Basic Usage

### Numerical Data Example

```ruby
require 'grydra'

data_input = [[170, 25], [160, 30], [180, 22]]
data_output = [[65], [60], [75]]
structures = [[4, 1], [3, 1]]

network = GRYDRA::EasyNetwork.new(print_epochs = true)

network.train_numerical(
  data_input,
  data_output,
  structures,
  learning_rate = 0.05,
  epochs = 15000,
  normalization = :max
)

new_data = [[172, 26]]
predictions = network.predict_numerical(new_data, :max)

puts "Predicted weight: #{predictions[0][0].round(2)} kg"
```

---

### Hash Data Example

```ruby
require 'grydra'

data_hash = [
  { height: 170, is_new: false, label: 0 },
  { height: 160, is_new: true, label: 1 },
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

new_hashes = [{ height: 175, is_new: true }]
predictions = network.predict_hashes(new_hashes, input_keys, :max)

puts "Prediction: #{predictions[0][0].round(3)}"
```

---

## 🧩 Key Components

| Component       | Description                                                         |
| --------------- | ------------------------------------------------------------------- |
| `EasyNetwork`   | High-level interface for training on numerical, hash, and text data |
| `MainNetwork`   | Multi-subnet architecture manager                                   |
| `NeuralNetwork` | Core neural network implementation                                  |
| `DenseLayer`    | Fully connected layer                                               |
| `Neuron`        | Individual neuron representation                                    |
| `AdamOptimizer` | Optimizer implementation                                            |

---

## 🛠 Helper Functions

* `GRYDRA.save_model(model, name, path, vocabulary)`
* `GRYDRA.load_model(name, path)`
* `GRYDRA.save_vocabulary(vocabulary, name, path)`
* `GRYDRA.load_vocabulary(name, path)`
* `GRYDRA.describe_method(class_name, method_name)`
* `GRYDRA.list_methods_available()`
* `GRYDRA.generate_example(num, filename, ext, path)`

---

## 📂 Examples

Use the example generator:

```ruby
GRYDRA.generate_example(1)
```

---

## 🧭 Project Structure

```
/GRYDRA
 ├── gems/
 │   ├── grydra-0.1.7.gem
 │   ├── grydra-0.1.8.gem
 │   ├── grydra-0.1.9.gem
 │   ├── grydra-0.2.0.gem
 │   └── grydra-1.0.0.gem
 ├── lib/
 │   └── gr/
 │       └── core.rb
 ├── README.md
 └── LICENCE
```

---

## 💙 Support Gabo-Razo
<center>
If this project has been useful, consider supporting **Gabo-Razo** via GitHub Sponsors.  
Your contribution helps keep development active and improve future releases.

</center>

---

<div align="center">

  <img 
    src="https://avatars.githubusercontent.com/u/219750358?v=4" 
    width="90" 
    style="border-radius: 50%; margin-bottom: 12px;"
  />

  <p style="font-size: 22px; font-weight: 800; margin: 0;">
    Gabo-Razo
  </p>

  <p>
    <strong>Developer • Ruby • Python • C++ • Java • Dart • COBOL</strong>
  </p>

  <p>
    <a href="https://github.com/Gabo-Razo?tab=followers">
      ⭐ Follow on GitHub
    </a>
  </p>

  <a href="https://github.com/sponsors/Gabo-Razo">
    <img 
      src="https://img.shields.io/badge/Sponsor_Me-FF4081?style=for-the-badge&logo=githubsponsors&logoColor=white"
      height="44"
    />
  </a>

</div>

---

## 📜 License

[GNU License](LICENCE)
