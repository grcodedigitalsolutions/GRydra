# GRYDRA v2.0 - Neural Networks for Ruby

![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Ruby](https://img.shields.io/badge/Ruby-2.7+-red)
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)

A complete, modular, and powerful neural network library for Ruby.

## Installation

```bash
gem install grydra
```

## Quick Start

```ruby
require 'grydra'

model = GRYDRA::Networks::EasyNetwork.new(print_epochs: true)
data_input = [[1, 2], [2, 3], [3, 4], [4, 5]]
data_output = [[3], [5], [7], [9]]

model.train_numerical(data_input, data_output, [[4, 1]], 0.1, 1000, :max)
predictions = model.predict_numerical([[5, 6]])
puts predictions  # => [[11.0]]
```

## Examples - From Simple to Complex

### Level 1: Basic Examples

#### Example 1.1: Simple Addition

```ruby
require 'grydra'

# Learn to add two numbers
model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)

data_input = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
data_output = [[2], [4], [6], [8], [10]]

model.train_numerical(data_input, data_output, [[3, 1]], 0.1, 500, :max)

# Test
result = model.predict_numerical([[6, 6]])
puts "6 + 6 = #{result[0][0].round(0)}"  # => 12
```

#### Example 1.2: Temperature Conversion (Celsius to Fahrenheit)

```ruby
require 'grydra'

model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)

# Training data: Celsius -> Fahrenheit
celsius = [[0], [10], [20], [30], [40], [100]]
fahrenheit = [[32], [50], [68], [86], [104], [212]]

model.train_numerical(celsius, fahrenheit, [[3, 1]], 0.1, 1000, :max)

# Convert 25°C to Fahrenheit
result = model.predict_numerical([[25]])
puts "25°C = #{result[0][0].round(1)}°F"  # => ~77°F
```

#### Example 1.3: Simple Classification (Pass/Fail)

```ruby
require 'grydra'

model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)

# Student scores -> Pass (1) or Fail (0)
scores = [[45], [55], [65], [75], [85], [35], [95], [40]]
results = [[0], [1], [1], [1], [1], [0], [1], [0]]

model.train_numerical(scores, results, [[3, 1]], 0.1, 1000, :max)

# Predict for score 70
prediction = model.predict_numerical([[70]])
pass_fail = prediction[0][0] > 0.5 ? "PASS" : "FAIL"
puts "Score 70: #{pass_fail}"
```

### Level 2: Intermediate Examples

#### Example 2.1: House Price Prediction

```ruby
require 'grydra'

model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)
model.configure_adam_optimizer(alpha: 0.001)

# [sqft, bedrooms, age]
houses = [
  [1200, 2, 5], [1500, 3, 3], [1800, 3, 8],
  [2000, 4, 2], [2200, 4, 10], [1000, 2, 15]
]
prices = [[250000], [300000], [280000], [350000], [320000], [200000]]

model.train_numerical(houses, prices, [[6, 4, 1]], 0.05, 2000, :max,
                     lambda_l2: 0.001, patience: 100)

# Predict price for 1600 sqft, 3 bed, 4 years old
new_house = [[1600, 3, 4]]
price = model.predict_numerical(new_house)
puts "Predicted price: $#{price[0][0].round(0)}"
```


#### Example 2.2: Customer Churn Prediction

```ruby
require 'grydra'

model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)

# [age, monthly_spend, months_active, support_tickets]
customers = [
  [25, 50, 3, 5], [45, 200, 24, 0], [35, 100, 12, 2],
  [55, 180, 36, 1], [28, 40, 2, 8], [50, 190, 30, 0]
]
churn = [[1], [0], [0], [0], [1], [0]]  # 1=will churn, 0=will stay

model.train_numerical(customers, churn, [[6, 4, 1]], 0.1, 1500, :zscore,
                     dropout: true, dropout_rate: 0.3)

# Predict for new customer
new_customer = [[30, 75, 6, 3]]
probability = model.predict_numerical(new_customer, :zscore)
risk = probability[0][0] > 0.5 ? "HIGH RISK" : "LOW RISK"
puts "Churn probability: #{(probability[0][0] * 100).round(1)}% - #{risk}"
```

#### Example 2.3: Sentiment Analysis

```ruby
require 'grydra'

model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)

texts = [
  "I love this product", "Amazing quality",
  "Terrible experience", "Waste of money",
  "Highly recommend", "Very disappointed",
  "Excellent service", "Poor quality"
]
sentiments = [[1], [1], [0], [0], [1], [0], [1], [0]]

model.train_text(texts, sentiments, [[8, 4, 1]], 0.1, 1000, :max,
                lambda_l1: 0.001, dropout: true, dropout_rate: 0.2)

# Analyze new reviews
new_reviews = ["Best purchase ever", "Complete garbage"]
predictions = model.predict_text(new_reviews, :max)

new_reviews.each_with_index do |review, i|
  score = predictions[i][0]
  sentiment = score > 0.5 ? "POSITIVE" : "NEGATIVE"
  puts "\"#{review}\" => #{sentiment}"
end
```

### Level 3: Advanced Examples

#### Example 3.1: Multi-Output Prediction (Weather)

```ruby
require 'grydra'

model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)

# [hour, latitude, longitude] -> [temperature, humidity]
conditions = [
  [6, 40.7, -74.0], [12, 40.7, -74.0], [18, 40.7, -74.0],
  [6, 34.0, -118.2], [12, 34.0, -118.2], [18, 34.0, -118.2]
]
weather = [[15, 70], [25, 50], [20, 60], [18, 40], [30, 30], [25, 35]]

model.train_numerical(conditions, weather, [[6, 4, 2]], 0.05, 2000, :max)

# Predict weather at 2 PM in New York
prediction = model.predict_numerical([[14, 40.7, -74.0]])
puts "Temperature: #{prediction[0][0].round(1)}°C"
puts "Humidity: #{prediction[0][1].round(1)}%"
```

#### Example 3.2: Time Series Prediction (Stock Prices)

```ruby
require 'grydra'

model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)
model.configure_adam_optimizer(alpha: 0.001)

# Last 3 days -> Next day price
sequences = [
  [100, 102, 101], [102, 101, 103], [101, 103, 105],
  [103, 105, 104], [105, 104, 106], [104, 106, 108]
]
next_prices = [[103], [105], [104], [106], [108], [107]]

model.train_numerical(sequences, next_prices, [[6, 4, 1]], 0.01, 2000, :max,
                     lambda_l2: 0.01, patience: 100)

# Predict next price
last_3_days = [[106, 108, 107]]
prediction = model.predict_numerical(last_3_days)
puts "Predicted next price: $#{prediction[0][0].round(2)}"
```

#### Example 3.3: Image Classification (Simplified)

```ruby
require 'grydra'

model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)

# Simplified 4x4 "images" (16 pixels) -> 2 classes
images = [
  [1,1,0,0, 1,1,0,0, 0,0,0,0, 0,0,0,0],  # Pattern A
  [0,0,1,1, 0,0,1,1, 0,0,0,0, 0,0,0,0],  # Pattern B
  [1,1,0,0, 1,1,0,0, 0,0,0,0, 0,0,0,0],  # Pattern A
  [0,0,1,1, 0,0,1,1, 0,0,0,0, 0,0,0,0]   # Pattern B
]
labels = [[0], [1], [0], [1]]

model.train_numerical(images, labels, [[8, 4, 1]], 0.1, 1000, :max)

# Classify new image
new_image = [[1,1,0,0, 1,1,0,0, 0,0,0,0, 0,0,0,0]]
prediction = model.predict_numerical(new_image)
class_label = prediction[0][0] > 0.5 ? "Pattern B" : "Pattern A"
puts "Classification: #{class_label}"
```

### Level 4: Expert Examples

#### Example 4.1: Cross-Validation

```ruby
require 'grydra'

# Generate data
synthetic = GRYDRA::Preprocessing::Data.generate_synthetic_data(100, 3, 0.1, 42)
data_x = synthetic[:data]
data_y = synthetic[:labels]

# 5-fold cross-validation
result = GRYDRA::Training::CrossValidation.cross_validation(data_x, data_y, 5) do |train_x, train_y, test_x, test_y|
  model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)
  model.train_numerical(train_x, train_y, [[4, 3, 1]], 0.1, 500, :max, patience: 50)
  
  predictions = model.predict_numerical(test_x)
  GRYDRA::Metrics.mse(predictions.flatten, test_y.flatten)
end

puts "Cross-Validation Results:"
puts "  Average Error: #{result[:average].round(6)}"
puts "  Std Deviation: #{result[:deviation].round(6)}"
puts "  Fold Errors: #{result[:errors].map { |e| e.round(4) }}"
```


#### Example 4.2: Hyperparameter Search

```ruby
require 'grydra'

data_x = [[1], [2], [3], [4], [5], [6], [7], [8]]
data_y = [[2], [4], [6], [8], [10], [12], [14], [16]]

param_grid = [
  { rate: 0.01, epochs: 800, lambda_l2: 0.001 },
  { rate: 0.05, epochs: 600, lambda_l2: 0.01 },
  { rate: 0.1, epochs: 500, lambda_l2: 0.001 }
]

best = GRYDRA::Training::HyperparameterSearch.hyperparameter_search(
  data_x, data_y, param_grid, verbose: true
) do |params, x, y|
  model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)
  model.train_numerical(x, y, [[3, 1]], params[:rate], params[:epochs], :max,
                       lambda_l2: params[:lambda_l2], patience: 50)
  
  predictions = model.predict_numerical(x)
  GRYDRA::Metrics.mse(predictions.flatten, y.flatten)
end

puts "\nBest parameters: #{best[:parameters]}"
puts "Best score: #{best[:score].round(6)}"
```

#### Example 4.3: PCA for Dimensionality Reduction

```ruby
require 'grydra'

# High-dimensional data (5 features)
data = [
  [2.5, 2.4, 1.2, 0.8, 3.1],
  [0.5, 0.7, 0.3, 0.2, 0.9],
  [2.2, 2.9, 1.5, 1.0, 2.8],
  [1.9, 2.2, 1.0, 0.7, 2.5],
  [3.1, 3.0, 1.8, 1.2, 3.5]
]

# Reduce to 2 dimensions
pca_result = GRYDRA::Preprocessing::PCA.pca(data, components: 2)

puts "PCA Results:"
puts "  Explained Variance: #{pca_result[:explained_variance].map { |v| (v * 100).round(2) }}%"
puts "  Eigenvalues: #{pca_result[:eigenvalues].map { |v| v.round(4) }}"

# Transform new data
new_data = [[2.0, 2.5, 1.1, 0.8, 2.7]]
transformed = GRYDRA::Preprocessing::PCA.transform(new_data, pca_result)
puts "  Transformed: #{transformed[0].map { |v| v.round(3) }}"
```

#### Example 4.4: Ensemble with Multiple Subnets

```ruby
require 'grydra'

# Create ensemble network
network = GRYDRA::Networks::MainNetwork.new(print_epochs: false)

# Add multiple subnets with different architectures
network.add_subnet([2, 4, 1], [:tanh, :sigmoid])
network.add_subnet([2, 3, 1], [:relu, :tanh])
network.add_subnet([2, 5, 1], [:sigmoid, :sigmoid])

# XOR problem
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [1], [1], [0]]

# Train all subnets
network.train_subnets(
  [
    {input: inputs, output: outputs},
    {input: inputs, output: outputs},
    {input: inputs, output: outputs}
  ],
  0.5, 3000, patience: 150
)

# Test with ensemble
puts "XOR Results (Ensemble):"
inputs.each do |input|
  output = network.combine_results(input)
  result = output[0] > 0.5 ? 1 : 0
  puts "  #{input} => #{output[0].round(3)} (#{result})"
end
```

#### Example 4.5: Save and Load Models

```ruby
require 'grydra'

# Train model
model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)
data_input = [[1, 2], [2, 3], [3, 4], [4, 5]]
data_output = [[3], [5], [7], [9]]
model.train_numerical(data_input, data_output, [[4, 1]], 0.1, 1000, :max)

# Save model
GRYDRA::Utils::Persistence.save_model(model, "my_model", "./models")

# Later... load and use
loaded_model = GRYDRA::Utils::Persistence.load_model("my_model", "./models")
prediction = loaded_model.predict_numerical([[5, 6]])
puts "Prediction: #{prediction[0][0].round(2)}"

# Show model summary
GRYDRA::Utils::Persistence.summary_model(loaded_model)
```

#### Example 4.6: Using Different Loss Functions

```ruby
require 'grydra'

predictions = [0.9, 0.2, 0.8, 0.1, 0.95]
targets = [1, 0, 1, 0, 1]

puts "Loss Function Comparison:"
puts "  MSE: #{GRYDRA::Losses.mse(predictions, targets).round(4)}"
puts "  MAE: #{GRYDRA::Losses.mae(predictions, targets).round(4)}"
puts "  Huber: #{GRYDRA::Losses.huber(predictions, targets, delta: 1.0).round(4)}"
puts "  Binary Cross-Entropy: #{GRYDRA::Losses.binary_crossentropy(predictions, targets).round(4)}"
puts "  Log-Cosh: #{GRYDRA::Losses.log_cosh(predictions, targets).round(4)}"
```

#### Example 4.7: Network Visualization

```ruby
require 'grydra'

model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)
data_input = [[1, 2], [2, 3], [3, 4]]
data_output = [[3], [5], [7]]
model.train_numerical(data_input, data_output, [[4, 3, 1]], 0.1, 500, :max)

# Visualize architecture
GRYDRA::Utils::Visualization.plot_architecture_ascii(model.network)

# Analyze gradients
analysis = GRYDRA::Utils::Visualization.analyze_gradients(model.network)
puts "\nGradient Analysis:"
puts "  Average: #{analysis[:average].round(6)}"
puts "  Max: #{analysis[:maximum].round(6)}"
puts "  Min: #{analysis[:minimum].round(6)}"
puts "  Total Parameters: #{analysis[:total_parameters]}"
```

### Level 5: Expert Advanced Examples

#### Example 5.1: Custom Network with Different Optimizers

```ruby
require 'grydra'

# Compare different optimizers on the same problem
data_x = Array.new(50) { |i| [i / 10.0] }
data_y = data_x.map { |x| [Math.sin(x[0]) * 10 + 5] }

optimizers = {
  'Adam' => GRYDRA::Optimizers::Adam.new(alpha: 0.01),
  'SGD with Momentum' => GRYDRA::Optimizers::SGD.new(learning_rate: 0.1, momentum: 0.9),
  'RMSprop' => GRYDRA::Optimizers::RMSprop.new(learning_rate: 0.01),
  'AdaGrad' => GRYDRA::Optimizers::AdaGrad.new(learning_rate: 0.1),
  'AdamW' => GRYDRA::Optimizers::AdamW.new(alpha: 0.01, weight_decay: 0.01)
}

optimizers.each do |name, optimizer|
  network = GRYDRA::Networks::NeuralNetwork.new([1, 8, 8, 1], activations: [:relu, :relu, :tanh])
  network.instance_variable_set(:@optimizer, optimizer)
  
  network.train(data_x, data_y, 0.01, 500, patience: 100)
  
  test_x = [[2.5]]
  prediction = network.calculate_outputs(test_x[0])
  actual = Math.sin(2.5) * 10 + 5
  error = (prediction[0] - actual).abs
  
  puts "#{name}: Prediction=#{prediction[0].round(3)}, Actual=#{actual.round(3)}, Error=#{error.round(3)}"
end
```

#### Example 5.2: Using Callbacks for Advanced Training Control

```ruby
require 'grydra'

# Create network
network = GRYDRA::Networks::NeuralNetwork.new([2, 6, 4, 1], 
                                              print_epochs: true,
                                              activations: [:relu, :relu, :sigmoid])

# XOR problem
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [1], [1], [0]]

# Setup callbacks
early_stopping = GRYDRA::Callbacks::EarlyStopping.new(
  monitor: :loss,
  patience: 50,
  min_delta: 0.001,
  restore_best: true
)

lr_scheduler = GRYDRA::Callbacks::LearningRateScheduler.new do |epoch, current_lr|
  # Reduce learning rate every 100 epochs
  epoch % 100 == 0 && epoch > 0 ? current_lr * 0.9 : current_lr
end

reduce_lr = GRYDRA::Callbacks::ReduceLROnPlateau.new(
  monitor: :loss,
  factor: 0.5,
  patience: 30,
  min_lr: 1e-6
)

csv_logger = GRYDRA::Callbacks::CSVLogger.new('training_log.csv')

# Train with callbacks (manual implementation for demonstration)
network.train(inputs, outputs, 0.5, 1000, patience: 100)

puts "\n✅ Training completed with callbacks"
puts "Check 'training_log.csv' for detailed logs"
```

#### Example 5.3: Building Custom Architecture with Layers

```ruby
require 'grydra'

# Build a custom network manually using layers
class CustomNetwork
  attr_accessor :layers
  
  def initialize
    @layers = []
    # Input: 10 features
    @layers << GRYDRA::Layers::Dense.new(16, 10, :relu)
    @layers << GRYDRA::Layers::Dense.new(8, 16, :leaky_relu)
    @layers << GRYDRA::Layers::Dense.new(4, 8, :swish)
    @layers << GRYDRA::Layers::Dense.new(1, 4, :sigmoid)
  end
  
  def forward(input, dropout: false, dropout_rate: 0.3)
    output = input
    @layers.each_with_index do |layer, idx|
      # Apply dropout to hidden layers only
      apply_drop = dropout && idx < @layers.size - 1
      output = layer.calculate_outputs(output, apply_drop, dropout_rate)
    end
    output
  end
end

# Create and test custom network
custom_net = CustomNetwork.new
input = Array.new(10) { rand }
output = custom_net.forward(input, dropout: true)

puts "Custom Network Output: #{output.map { |v| v.round(4) }}"
puts "Network has #{custom_net.layers.size} layers"
custom_net.layers.each_with_index do |layer, i|
  puts "  Layer #{i + 1}: #{layer.neurons.size} neurons, activation: #{layer.activation}"
end
```

#### Example 5.4: Multi-Task Learning with Shared Layers

```ruby
require 'grydra'

# Simulate multi-task learning: predict both price and category
# [sqft, bedrooms, location_score] -> [price, category]

houses = [
  [1200, 2, 0.7], [1500, 3, 0.8], [1800, 3, 0.9],
  [2000, 4, 0.85], [2200, 4, 0.95], [1000, 2, 0.6]
]

# Task 1: Price (regression)
prices = [[250], [300], [280], [350], [320], [200]]

# Task 2: Category (classification: 0=budget, 1=luxury)
categories = [[0], [0], [1], [1], [1], [0]]

# Create two networks with similar architecture
price_model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)
category_model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)

# Train both tasks
puts "Training price prediction model..."
price_model.train_numerical(houses, prices, [[8, 6, 1]], 0.05, 1500, :zscore, lambda_l2: 0.001)

puts "Training category classification model..."
category_model.train_numerical(houses, categories, [[8, 4, 1]], 0.1, 1500, :zscore, dropout: true)

# Predict on new house
new_house = [[1600, 3, 0.75]]
predicted_price = price_model.predict_numerical(new_house, :zscore)
predicted_category = category_model.predict_numerical(new_house, :zscore)

puts "\n🏠 Multi-Task Prediction:"
puts "  Predicted Price: $#{predicted_price[0][0].round(0)}k"
puts "  Category: #{predicted_category[0][0] > 0.5 ? 'Luxury' : 'Budget'} (#{(predicted_category[0][0] * 100).round(1)}%)"
```

#### Example 5.5: Time Series with LSTM Layer

```ruby
require 'grydra'

# Create LSTM layer for sequence processing
lstm = GRYDRA::Layers::LSTM.new(units: 8, inputs_per_unit: 3, return_sequences: false)

# Time series data: [day1, day2, day3] -> predict day4
sequences = [
  [[100], [102], [101]],
  [[102], [101], [103]],
  [[101], [103], [105]],
  [[103], [105], [104]]
]

puts "LSTM Layer Processing:"
sequences.each_with_index do |seq, i|
  lstm.reset_state  # Reset for each sequence
  output = lstm.calculate_outputs(seq)
  puts "  Sequence #{i + 1}: #{seq.flatten} => Hidden State: #{output.map { |v| v.round(3) }}"
end
```

#### Example 5.6: Advanced Ensemble with Weighted Voting

```ruby
require 'grydra'

# Create ensemble with different architectures and activations
ensemble = GRYDRA::Networks::MainNetwork.new(print_epochs: false)

# Add diverse subnets
ensemble.add_subnet([2, 8, 4, 1], [:relu, :relu, :sigmoid])
ensemble.add_subnet([2, 6, 3, 1], [:tanh, :tanh, :tanh])
ensemble.add_subnet([2, 10, 5, 1], [:leaky_relu, :swish, :sigmoid])
ensemble.add_subnet([2, 4, 4, 1], [:gelu, :relu, :sigmoid])

# XOR problem
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [1], [1], [0]]

# Train all subnets
data = Array.new(4) { { input: inputs, output: outputs } }
ensemble.train_subnets(data, 0.7, 2000, patience: 200)

# Test with different voting strategies
puts "\n🎯 Ensemble Predictions:"
inputs.each do |input|
  # Average voting
  avg_output = ensemble.combine_results(input)
  
  # Weighted voting (give more weight to better performing models)
  weights = [0.4, 0.3, 0.2, 0.1]
  weighted_output = ensemble.combine_results_weighted(input, weights)
  
  puts "Input: #{input}"
  puts "  Average: #{avg_output[0].round(3)} => #{avg_output[0] > 0.5 ? 1 : 0}"
  puts "  Weighted: #{weighted_output[0].round(3)} => #{weighted_output[0] > 0.5 ? 1 : 0}"
end
```

#### Example 5.7: Real-World: Credit Card Fraud Detection

```ruby
require 'grydra'

# Simulate credit card transaction data
# [amount, time_of_day, merchant_category, distance_from_home, frequency]
transactions = [
  [50, 14, 1, 2, 5],      # Normal
  [30, 10, 2, 1, 8],      # Normal
  [5000, 3, 5, 500, 1],   # Fraud
  [100, 18, 3, 5, 6],     # Normal
  [2000, 2, 4, 300, 1],   # Fraud
  [75, 12, 1, 3, 7],      # Normal
  [3500, 4, 5, 450, 1],   # Fraud
  [45, 16, 2, 2, 9]       # Normal
]

labels = [[0], [0], [1], [0], [1], [0], [1], [0]]  # 0=normal, 1=fraud

# Create model with class imbalance handling
model = GRYDRA::Networks::EasyNetwork.new(print_epochs: true)
model.configure_adam_optimizer(alpha: 0.001)

# Train with heavy regularization and dropout
model.train_numerical(transactions, labels, [[10, 8, 4, 1]], 0.05, 2000, :zscore,
                     lambda_l1: 0.001, lambda_l2: 0.01,
                     dropout: true, dropout_rate: 0.4,
                     patience: 150)

# Test on new transactions
new_transactions = [
  [60, 15, 1, 2, 6],      # Should be normal
  [4000, 3, 5, 400, 1]    # Should be fraud
]

predictions = model.predict_numerical(new_transactions, :zscore)

puts "\n💳 Fraud Detection Results:"
new_transactions.each_with_index do |trans, i|
  prob = predictions[i][0]
  status = prob > 0.5 ? "🚨 FRAUD" : "✅ NORMAL"
  puts "Transaction #{i + 1}: #{trans}"
  puts "  Fraud Probability: #{(prob * 100).round(2)}% - #{status}"
end
```

#### Example 5.8: A/B Testing with Statistical Validation

```ruby
require 'grydra'

# Simulate A/B test data: [page_views, time_on_site, clicks] -> conversion
variant_a = [
  [100, 120, 5], [150, 180, 8], [80, 90, 3],
  [120, 150, 6], [90, 100, 4]
]
conversions_a = [[0], [1], [0], [1], [0]]

variant_b = [
  [110, 140, 7], [160, 200, 10], [85, 110, 5],
  [130, 170, 8], [95, 120, 6]
]
conversions_b = [[1], [1], [0], [1], [1]]

# Train separate models
model_a = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)
model_b = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)

model_a.train_numerical(variant_a, conversions_a, [[6, 4, 1]], 0.1, 1000, :max)
model_b.train_numerical(variant_b, conversions_b, [[6, 4, 1]], 0.1, 1000, :max)

# Cross-validate both variants
puts "📊 A/B Test Results:"

result_a = GRYDRA::Training::CrossValidation.cross_validation(variant_a, conversions_a, 3) do |train_x, train_y, test_x, test_y|
  m = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)
  m.train_numerical(train_x, train_y, [[6, 4, 1]], 0.1, 500, :max)
  preds = m.predict_numerical(test_x)
  GRYDRA::Metrics.mse(preds.flatten, test_y.flatten)
end

result_b = GRYDRA::Training::CrossValidation.cross_validation(variant_b, conversions_b, 3) do |train_x, train_y, test_x, test_y|
  m = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)
  m.train_numerical(train_x, train_y, [[6, 4, 1]], 0.1, 500, :max)
  preds = m.predict_numerical(test_x)
  GRYDRA::Metrics.mse(preds.flatten, test_y.flatten)
end

puts "Variant A - Avg Error: #{result_a[:average].round(4)}, StdDev: #{result_a[:deviation].round(4)}"
puts "Variant B - Avg Error: #{result_b[:average].round(4)}, StdDev: #{result_b[:deviation].round(4)}"
puts "Winner: #{result_b[:average] < result_a[:average] ? 'Variant B' : 'Variant A'}"
```

#### Example 5.9: Product Recommendation System

```ruby
require 'grydra'

# User features: [age, purchase_history, browsing_time, category_preference]
# Product features: [price_range, popularity, category_match]
# Combined: [user_features + product_features] -> purchase_probability

training_data = [
  [25, 5, 120, 0.8, 1, 0.7, 0.9],   # Young user, low price, high match -> buy
  [45, 20, 200, 0.6, 3, 0.9, 0.7],  # Mature user, high price, popular -> buy
  [30, 2, 50, 0.3, 1, 0.3, 0.2],    # Low engagement, low match -> no buy
  [50, 15, 180, 0.9, 2, 0.8, 0.95], # High engagement, good match -> buy
  [22, 1, 30, 0.2, 1, 0.2, 0.1],    # New user, poor match -> no buy
  [35, 10, 150, 0.7, 2, 0.6, 0.8]   # Average user, decent match -> buy
]

labels = [[1], [1], [0], [1], [0], [1]]

# Train recommendation model
model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)
model.configure_adam_optimizer(alpha: 0.001)

model.train_numerical(training_data, labels, [[12, 8, 4, 1]], 0.05, 2000, :zscore,
                     lambda_l2: 0.001, dropout: true, dropout_rate: 0.2,
                     patience: 150)

# Recommend products for new user-product pairs
new_pairs = [
  [28, 3, 90, 0.5, 1, 0.5, 0.6],   # Young user, budget product
  [40, 12, 160, 0.8, 3, 0.85, 0.9] # Mature user, premium product
]

predictions = model.predict_numerical(new_pairs, :zscore)

puts "\n🛍️ Product Recommendations:"
new_pairs.each_with_index do |pair, i|
  prob = predictions[i][0]
  recommendation = prob > 0.5 ? "✅ RECOMMEND" : "❌ DON'T RECOMMEND"
  puts "User-Product Pair #{i + 1}:"
  puts "  Purchase Probability: #{(prob * 100).round(2)}%"
  puts "  Decision: #{recommendation}"
end
```

#### Example 5.10: Anomaly Detection in IoT Sensor Data

```ruby
require 'grydra'

# IoT sensor readings: [temperature, humidity, pressure, vibration, power_consumption]
# Normal operating conditions
normal_data = [
  [22.5, 45, 1013, 0.2, 150],
  [23.0, 47, 1012, 0.3, 155],
  [22.8, 46, 1013, 0.25, 152],
  [23.2, 48, 1011, 0.28, 158],
  [22.6, 45, 1014, 0.22, 151],
  [23.1, 47, 1012, 0.29, 156]
]

# Anomalous conditions
anomaly_data = [
  [35.0, 80, 980, 2.5, 300],   # Overheating
  [15.0, 20, 1050, 0.1, 50],   # Underpowered
  [25.0, 50, 1010, 5.0, 200],  # High vibration
  [40.0, 90, 970, 3.0, 350]    # Multiple issues
]

# Label data: 0 = normal, 1 = anomaly
all_data = normal_data + anomaly_data
labels = [[0]] * normal_data.size + [[1]] * anomaly_data.size

# Train anomaly detection model
model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)
model.configure_adam_optimizer(alpha: 0.001)

model.train_numerical(all_data, labels, [[10, 8, 4, 1]], 0.05, 2000, :zscore,
                     lambda_l1: 0.001, lambda_l2: 0.01,
                     dropout: true, dropout_rate: 0.3,
                     patience: 150)

# Monitor new sensor readings
new_readings = [
  [22.9, 46, 1013, 0.26, 153],  # Should be normal
  [38.0, 85, 975, 2.8, 320],    # Should be anomaly
  [23.5, 49, 1011, 0.31, 159]   # Should be normal
]

predictions = model.predict_numerical(new_readings, :zscore)

puts "\n🔍 IoT Anomaly Detection:"
new_readings.each_with_index do |reading, i|
  prob = predictions[i][0]
  status = prob > 0.5 ? "⚠️ ANOMALY DETECTED" : "✅ NORMAL"
  puts "Sensor Reading #{i + 1}: #{reading}"
  puts "  Anomaly Score: #{(prob * 100).round(2)}%"
  puts "  Status: #{status}"
end
```

#### Example 5.11: Dynamic Pricing Optimization

```ruby
require 'grydra'

# Features: [demand, competitor_price, inventory_level, day_of_week, season, customer_segment]
# Target: optimal_price_multiplier (0.8 to 1.5)

pricing_data = [
  [100, 50, 200, 1, 1, 1],  # High demand, weekday, winter, regular -> 1.2x
  [50, 45, 500, 6, 2, 2],   # Low demand, weekend, spring, premium -> 0.9x
  [150, 55, 50, 5, 3, 1],   # Very high demand, low stock, summer -> 1.4x
  [80, 48, 300, 3, 1, 2],   # Medium demand, good stock -> 1.0x
  [30, 40, 600, 7, 4, 3],   # Low demand, high stock, fall, budget -> 0.8x
  [120, 52, 100, 2, 3, 1]   # High demand, low stock, summer -> 1.3x
]

price_multipliers = [[1.2], [0.9], [1.4], [1.0], [0.8], [1.3]]

# Train pricing model
model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)
model.configure_adam_optimizer(alpha: 0.001)

model.train_numerical(pricing_data, price_multipliers, [[10, 8, 4, 1]], 0.05, 2000, :zscore,
                     lambda_l2: 0.001, patience: 150)

# Optimize pricing for new scenarios
new_scenarios = [
  [110, 51, 150, 4, 3, 1],  # High demand, Thursday, summer
  [40, 43, 450, 6, 2, 3],   # Low demand, Saturday, spring, budget
  [90, 49, 250, 2, 1, 2]    # Medium demand, Tuesday, winter, premium
]

predictions = model.predict_numerical(new_scenarios, :zscore)

puts "\n💰 Dynamic Pricing Recommendations:"
base_price = 100
new_scenarios.each_with_index do |scenario, i|
  multiplier = predictions[i][0]
  optimal_price = base_price * multiplier
  
  puts "Scenario #{i + 1}: Demand=#{scenario[0]}, Competitor=$#{scenario[1]}, Stock=#{scenario[2]}"
  puts "  Price Multiplier: #{multiplier.round(2)}x"
  puts "  Optimal Price: $#{optimal_price.round(2)} (base: $#{base_price})"
  puts "  Strategy: #{multiplier > 1.1 ? 'Premium Pricing' : (multiplier < 0.95 ? 'Discount Pricing' : 'Standard Pricing')}"
end
```

#### Example 5.12: Medical Diagnosis Assistant

```ruby
require 'grydra'

# Patient features: [age, blood_pressure, cholesterol, glucose, bmi, family_history, smoking]
# Diagnosis: 0 = healthy, 1 = at risk

patient_data = [
  [45, 120, 180, 90, 22, 0, 0],   # Healthy
  [55, 140, 240, 110, 28, 1, 1],  # At risk
  [38, 115, 170, 85, 21, 0, 0],   # Healthy
  [62, 150, 260, 125, 31, 1, 1],  # At risk
  [50, 135, 220, 105, 27, 1, 0],  # At risk
  [42, 118, 175, 88, 23, 0, 0],   # Healthy
  [58, 145, 250, 120, 30, 1, 1],  # At risk
  [40, 122, 185, 92, 24, 0, 0]    # Healthy
]

diagnoses = [[0], [1], [0], [1], [1], [0], [1], [0]]

# Train diagnostic model with high accuracy requirements
model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)
model.configure_adam_optimizer(alpha: 0.001)

model.train_numerical(patient_data, diagnoses, [[12, 10, 6, 1]], 0.05, 3000, :zscore,
                     lambda_l2: 0.001, dropout: true, dropout_rate: 0.2,
                     patience: 200)

# Evaluate model performance
metrics = model.evaluate_model(patient_data, diagnoses, [:accuracy, :confusion_matrix], :zscore)

puts "\n🏥 Medical Diagnosis Model Performance:"
puts "  Accuracy: #{(metrics[:accuracy] * 100).round(2)}%"
if metrics[:confusion_matrix]
  cm = metrics[:confusion_matrix]
  puts "  True Positives: #{cm[:tp]}"
  puts "  True Negatives: #{cm[:tn]}"
  puts "  False Positives: #{cm[:fp]}"
  puts "  False Negatives: #{cm[:fn]}"
end

# Diagnose new patients
new_patients = [
  [48, 125, 195, 95, 25, 0, 0],   # Borderline
  [60, 155, 270, 130, 32, 1, 1]   # High risk
]

predictions = model.predict_numerical(new_patients, :zscore)

puts "\n👨‍⚕️ New Patient Diagnoses:"
new_patients.each_with_index do |patient, i|
  risk_score = predictions[i][0]
  risk_level = risk_score > 0.7 ? "HIGH RISK" : (risk_score > 0.3 ? "MODERATE RISK" : "LOW RISK")
  
  puts "Patient #{i + 1}: Age=#{patient[0]}, BP=#{patient[1]}, Cholesterol=#{patient[2]}"
  puts "  Risk Score: #{(risk_score * 100).round(2)}%"
  puts "  Assessment: #{risk_level}"
  puts "  Recommendation: #{risk_score > 0.5 ? 'Immediate consultation recommended' : 'Regular monitoring'}"
end
```

## Features

### Core Components
- **7 Activation Functions**: tanh, relu, sigmoid, leaky_relu, swish, gelu, softmax
- **8 Loss Functions**: MSE, MAE, Huber, Binary/Categorical Cross-Entropy, Hinge, Log-Cosh, Quantile
- **5 Optimizers**: Adam, SGD (with momentum/Nesterov), RMSprop, AdaGrad, AdamW
- **3 Layer Types**: Dense (fully connected), Convolutional, LSTM
- **6 Training Callbacks**: EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, ProgressBar

### Advanced Features
- **Regularization**: L1 (Lasso), L2 (Ridge), Dropout
- **Normalization**: Z-score, Min-Max, Feature-wise
- **Model Validation**: K-fold cross-validation, train/test split
- **Hyperparameter Tuning**: Grid search with parallel evaluation
- **Dimensionality Reduction**: PCA with power iteration
- **Text Processing**: Vocabulary creation, TF-IDF vectorization
- **Model Persistence**: Save/load models and vocabularies
- **Visualization**: ASCII architecture plots, gradient analysis, training curves
- **Ensemble Learning**: Multiple subnets with weighted voting

### Network Architectures
- **EasyNetwork**: High-level API for quick prototyping
- **MainNetwork**: Ensemble of multiple subnets
- **NeuralNetwork**: Low-level customizable architecture
- **Custom Layers**: Build your own layer types

### Level 6: Production-Ready Examples

#### Example 6.1: Email Spam Classifier

```ruby
require 'grydra'

# Email features: word frequencies and metadata
emails = [
  "free money click here win prize",
  "meeting tomorrow at 3pm",
  "congratulations you won lottery",
  "project deadline next week",
  "claim your prize now limited time",
  "lunch with team on friday"
]

labels = [[1], [0], [1], [0], [1], [0]]  # 1=spam, 0=not spam

model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)
model.train_text(emails, labels, [[10, 6, 1]], 0.1, 1500, :max,
                lambda_l1: 0.001, dropout: true, dropout_rate: 0.3)

# Test on new emails
new_emails = [
  "urgent meeting tomorrow morning",
  "click here for free money now"
]

predictions = model.predict_text(new_emails, :max)
new_emails.each_with_index do |email, i|
  spam_prob = predictions[i][0]
  label = spam_prob > 0.5 ? "📧 SPAM" : "✅ LEGITIMATE"
  puts "\"#{email}\""
  puts "  #{label} (#{(spam_prob * 100).round(1)}% confidence)\n\n"
end
```

#### Example 6.2: Employee Attrition Prediction

```ruby
require 'grydra'

# Employee data: [satisfaction, evaluation, projects, hours, tenure, accident, promotion]
employees = [
  [0.38, 0.53, 2, 157, 3, 0, 0],  # Left
  [0.80, 0.86, 5, 262, 6, 0, 0],  # Stayed
  [0.11, 0.88, 7, 272, 4, 0, 0],  # Left
  [0.72, 0.87, 5, 223, 5, 0, 0],  # Stayed
  [0.37, 0.52, 2, 159, 3, 0, 0],  # Left
  [0.41, 0.50, 2, 153, 3, 0, 0],  # Left
  [0.10, 0.77, 6, 247, 4, 0, 0],  # Left
  [0.92, 0.85, 5, 259, 5, 0, 1]   # Stayed
]

attrition = [[1], [0], [1], [0], [1], [1], [1], [0]]

model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)
model.configure_adam_optimizer(alpha: 0.001)

model.train_numerical(employees, attrition, [[12, 8, 4, 1]], 0.05, 2000, :zscore,
                     lambda_l2: 0.001, dropout: true, dropout_rate: 0.25,
                     patience: 150)

# Predict for new employees
new_employees = [
  [0.45, 0.55, 3, 180, 3, 0, 0],  # Moderate risk
  [0.85, 0.90, 4, 240, 6, 0, 1]   # Low risk
]

predictions = model.predict_numerical(new_employees, :zscore)

puts "👔 Employee Attrition Predictions:"
new_employees.each_with_index do |emp, i|
  risk = predictions[i][0]
  status = risk > 0.6 ? "🔴 HIGH RISK" : (risk > 0.3 ? "🟡 MODERATE" : "🟢 LOW RISK")
  puts "Employee #{i + 1}: Satisfaction=#{emp[0]}, Evaluation=#{emp[1]}, Projects=#{emp[2]}"
  puts "  Attrition Risk: #{(risk * 100).round(1)}% - #{status}"
  puts "  Action: #{risk > 0.5 ? 'Schedule retention interview' : 'Continue monitoring'}\n\n"
end
```

#### Example 6.3: Real Estate Valuation with Multiple Features

```ruby
require 'grydra'

# Property features: [sqft, bedrooms, bathrooms, age, lot_size, garage, pool, school_rating]
properties = [
  [1500, 3, 2, 10, 5000, 2, 0, 8],
  [2200, 4, 3, 5, 7500, 2, 1, 9],
  [1800, 3, 2, 15, 6000, 1, 0, 7],
  [2800, 5, 4, 3, 10000, 3, 1, 10],
  [1200, 2, 1, 25, 4000, 1, 0, 6],
  [2000, 4, 2.5, 8, 6500, 2, 0, 8],
  [3200, 5, 4, 2, 12000, 3, 1, 9],
  [1600, 3, 2, 12, 5500, 2, 0, 7]
]

prices = [[320], [485], [365], [650], [245], [410], [720], [340]]  # in thousands

model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)
model.configure_adam_optimizer(alpha: 0.001)

model.train_numerical(properties, prices, [[16, 12, 8, 1]], 0.03, 3000, :zscore,
                     lambda_l2: 0.001, patience: 200)

# Evaluate with cross-validation
cv_result = GRYDRA::Training::CrossValidation.cross_validation(properties, prices, 4) do |train_x, train_y, test_x, test_y|
  m = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)
  m.configure_adam_optimizer(alpha: 0.001)
  m.train_numerical(train_x, train_y, [[16, 12, 8, 1]], 0.03, 1500, :zscore, lambda_l2: 0.001, patience: 100)
  preds = m.predict_numerical(test_x, :zscore)
  GRYDRA::Metrics.mae(preds.flatten, test_y.flatten)
end

puts "🏡 Real Estate Model Performance:"
puts "  Average MAE: $#{cv_result[:average].round(2)}k"
puts "  Std Deviation: $#{cv_result[:deviation].round(2)}k\n\n"

# Predict new properties
new_properties = [
  [1900, 3, 2, 7, 6200, 2, 0, 8],
  [2500, 4, 3, 4, 8000, 2, 1, 9]
]

predictions = model.predict_numerical(new_properties, :zscore)

puts "Property Valuations:"
new_properties.each_with_index do |prop, i|
  price = predictions[i][0]
  puts "Property #{i + 1}: #{prop[0]} sqft, #{prop[1]} bed, #{prop[2]} bath"
  puts "  Estimated Value: $#{price.round(0)}k"
  puts "  Price per sqft: $#{(price * 1000 / prop[0]).round(2)}\n\n"
end
```

#### Example 6.4: Customer Lifetime Value Prediction

```ruby
require 'grydra'

# Customer features: [age, income, purchase_freq, avg_order, tenure_months, support_calls, returns]
customers = [
  [28, 45000, 12, 85, 24, 2, 1],
  [45, 95000, 24, 150, 48, 1, 0],
  [35, 65000, 18, 110, 36, 3, 2],
  [52, 120000, 30, 200, 60, 0, 0],
  [25, 35000, 6, 50, 12, 5, 3],
  [40, 80000, 20, 130, 42, 2, 1],
  [55, 110000, 28, 180, 54, 1, 0],
  [30, 50000, 10, 75, 18, 4, 2]
]

# Lifetime value in thousands
ltv = [[15], [45], [28], [65], [8], [35], [55], [12]]

model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)
model.configure_adam_optimizer(alpha: 0.001)

model.train_numerical(customers, ltv, [[14, 10, 6, 1]], 0.04, 2500, :zscore,
                     lambda_l2: 0.001, dropout: true, dropout_rate: 0.2,
                     patience: 180)

# Segment customers
new_customers = [
  [32, 55000, 15, 95, 30, 2, 1],   # Mid-value
  [48, 105000, 26, 175, 50, 1, 0], # High-value
  [26, 38000, 8, 60, 15, 4, 2]     # Low-value
]

predictions = model.predict_numerical(new_customers, :zscore)

puts "💰 Customer Lifetime Value Predictions:"
new_customers.each_with_index do |cust, i|
  value = predictions[i][0]
  segment = value > 40 ? "💎 PREMIUM" : (value > 20 ? "⭐ STANDARD" : "📊 BASIC")
  
  puts "Customer #{i + 1}: Age=#{cust[0]}, Income=$#{cust[1]}, Purchases/yr=#{cust[2]}"
  puts "  Predicted LTV: $#{value.round(1)}k"
  puts "  Segment: #{segment}"
  puts "  Strategy: #{value > 40 ? 'VIP treatment, exclusive offers' : (value > 20 ? 'Regular engagement, loyalty program' : 'Cost-effective retention')}\n\n"
end
```

#### Example 6.5: Network Traffic Anomaly Detection

```ruby
require 'grydra'

# Network metrics: [packets/sec, bytes/sec, connections, failed_logins, port_scans]
normal_traffic = [
  [1000, 500000, 50, 0, 0],
  [1200, 600000, 55, 1, 0],
  [950, 480000, 48, 0, 0],
  [1100, 550000, 52, 1, 0],
  [1050, 520000, 51, 0, 0]
]

attack_traffic = [
  [5000, 2500000, 200, 50, 10],  # DDoS
  [800, 400000, 45, 100, 0],     # Brute force
  [1500, 750000, 80, 5, 50],     # Port scan
  [10000, 5000000, 500, 20, 5]   # Combined attack
]

all_traffic = normal_traffic + attack_traffic
labels = [[0]] * normal_traffic.size + [[1]] * attack_traffic.size

model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)
model.configure_adam_optimizer(alpha: 0.001)

model.train_numerical(all_traffic, labels, [[10, 8, 4, 1]], 0.05, 2500, :zscore,
                     lambda_l1: 0.001, lambda_l2: 0.01,
                     dropout: true, dropout_rate: 0.35,
                     patience: 180)

# Monitor live traffic
live_traffic = [
  [1080, 540000, 53, 1, 0],      # Normal
  [4500, 2250000, 180, 45, 8],   # Suspicious
  [1150, 575000, 56, 0, 0]       # Normal
]

predictions = model.predict_numerical(live_traffic, :zscore)

puts "🔒 Network Security Monitoring:"
live_traffic.each_with_index do |traffic, i|
  threat_level = predictions[i][0]
  status = threat_level > 0.7 ? "🚨 CRITICAL" : (threat_level > 0.4 ? "⚠️ WARNING" : "✅ NORMAL")
  
  puts "Traffic Sample #{i + 1}: #{traffic[0]} pkt/s, #{traffic[2]} connections"
  puts "  Threat Score: #{(threat_level * 100).round(1)}%"
  puts "  Status: #{status}"
  puts "  Action: #{threat_level > 0.7 ? 'Block immediately, alert SOC' : (threat_level > 0.4 ? 'Increase monitoring' : 'Continue normal operation')}\n\n"
end
```

#### Example 6.6: Loan Default Prediction

```ruby
require 'grydra'

# Applicant features: [age, income, credit_score, debt_ratio, employment_years, loan_amount, previous_defaults]
applicants = [
  [35, 75000, 720, 0.3, 8, 25000, 0],   # Approved
  [28, 45000, 650, 0.5, 3, 15000, 1],   # Risky
  [42, 95000, 780, 0.2, 12, 35000, 0],  # Approved
  [25, 35000, 580, 0.7, 1, 10000, 2],   # Denied
  [50, 120000, 800, 0.15, 20, 50000, 0],# Approved
  [30, 50000, 620, 0.6, 4, 18000, 1],   # Risky
  [38, 85000, 750, 0.25, 10, 30000, 0], # Approved
  [26, 38000, 590, 0.65, 2, 12000, 2]   # Denied
]

defaults = [[0], [1], [0], [1], [0], [1], [0], [1]]

model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)
model.configure_adam_optimizer(alpha: 0.001)

model.train_numerical(applicants, defaults, [[14, 10, 6, 1]], 0.04, 2500, :zscore,
                     lambda_l2: 0.001, dropout: true, dropout_rate: 0.25,
                     patience: 180)

# Evaluate model
metrics = model.evaluate_model(applicants, defaults, [:accuracy, :confusion_matrix], :zscore)

puts "🏦 Loan Default Model Performance:"
puts "  Accuracy: #{(metrics[:accuracy] * 100).round(2)}%"
if metrics[:confusion_matrix]
  cm = metrics[:confusion_matrix]
  precision = GRYDRA::Metrics.precision(cm[:tp], cm[:fp])
  recall = GRYDRA::Metrics.recall(cm[:tp], cm[:fn])
  f1 = GRYDRA::Metrics.f1(precision, recall)
  puts "  Precision: #{(precision * 100).round(2)}%"
  puts "  Recall: #{(recall * 100).round(2)}%"
  puts "  F1 Score: #{(f1 * 100).round(2)}%\n\n"
end

# Evaluate new applications
new_applicants = [
  [33, 68000, 700, 0.35, 6, 22000, 0],
  [27, 42000, 610, 0.55, 2, 14000, 1]
]

predictions = model.predict_numerical(new_applicants, :zscore)

puts "Loan Application Decisions:"
new_applicants.each_with_index do |app, i|
  risk = predictions[i][0]
  decision = risk < 0.3 ? "✅ APPROVE" : (risk < 0.6 ? "⚠️ REVIEW" : "❌ DENY")
  
  puts "Applicant #{i + 1}: Age=#{app[0]}, Income=$#{app[1]}, Credit=#{app[2]}"
  puts "  Default Risk: #{(risk * 100).round(1)}%"
  puts "  Decision: #{decision}"
  puts "  Interest Rate: #{risk < 0.3 ? '5.5%' : (risk < 0.6 ? '8.5%' : 'N/A')}\n\n"
end
```

#### Example 6.7: Predictive Maintenance for Manufacturing

```ruby
require 'grydra'

# Machine sensor data: [temperature, vibration, pressure, rpm, power_consumption, runtime_hours, last_maintenance_days]
machine_data = [
  [65, 0.5, 100, 1500, 45, 1000, 30],   # Healthy
  [85, 2.5, 95, 1480, 52, 3500, 180],   # Needs maintenance
  [70, 0.8, 98, 1495, 46, 1500, 45],    # Healthy
  [95, 3.5, 88, 1450, 58, 4200, 240],   # Critical
  [68, 0.6, 99, 1498, 45, 1200, 35],    # Healthy
  [80, 2.0, 92, 1475, 50, 3000, 150],   # Needs maintenance
  [72, 1.0, 97, 1490, 47, 1800, 60],    # Healthy
  [90, 3.0, 90, 1460, 55, 3800, 210]    # Critical
]

maintenance_needed = [[0], [1], [0], [1], [0], [1], [0], [1]]

model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)
model.configure_adam_optimizer(alpha: 0.001)

model.train_numerical(machine_data, maintenance_needed, [[14, 10, 6, 1]], 0.04, 2500, :zscore,
                     lambda_l1: 0.001, lambda_l2: 0.01,
                     dropout: true, dropout_rate: 0.3,
                     patience: 180)

# Monitor machines
current_machines = [
  [73, 1.2, 96, 1488, 48, 2000, 75],    # Should be OK
  [88, 2.8, 91, 1465, 54, 3600, 195],   # Should need maintenance
  [67, 0.7, 99, 1497, 46, 1300, 40]     # Should be OK
]

predictions = model.predict_numerical(current_machines, :zscore)

puts "🏭 Predictive Maintenance System:"
current_machines.each_with_index do |machine, i|
  failure_risk = predictions[i][0]
  status = failure_risk > 0.7 ? "🔴 CRITICAL" : (failure_risk > 0.4 ? "🟡 WARNING" : "🟢 HEALTHY")
  
  puts "Machine #{i + 1}: Temp=#{machine[0]}°C, Vibration=#{machine[1]}mm/s, Runtime=#{machine[5]}hrs"
  puts "  Failure Risk: #{(failure_risk * 100).round(1)}%"
  puts "  Status: #{status}"
  puts "  Recommendation: #{failure_risk > 0.7 ? 'Schedule immediate maintenance' : (failure_risk > 0.4 ? 'Plan maintenance within 2 weeks' : 'Continue normal operation')}"
  puts "  Estimated Time to Failure: #{failure_risk > 0.7 ? '<1 week' : (failure_risk > 0.4 ? '2-4 weeks' : '>1 month')}\n\n"
end
```

#### Example 6.8: Sales Forecasting with Seasonality

```ruby
require 'grydra'

# Sales features: [month, day_of_week, is_holiday, temperature, marketing_spend, competitor_promo, inventory_level]
historical_sales = [
  [1, 1, 0, 35, 5000, 0, 1000],   # 45k sales
  [1, 5, 0, 32, 4500, 1, 950],    # 38k sales
  [2, 3, 1, 40, 8000, 0, 1200],   # 65k sales (Valentine's)
  [3, 6, 0, 55, 5500, 0, 1100],   # 48k sales
  [4, 2, 0, 65, 6000, 1, 1050],   # 42k sales
  [5, 7, 1, 75, 7000, 0, 1300],   # 58k sales (Memorial Day)
  [6, 4, 0, 85, 5000, 0, 1000],   # 50k sales
  [7, 1, 1, 90, 9000, 0, 1400]    # 72k sales (July 4th)
]

sales = [[45], [38], [65], [48], [42], [58], [50], [72]]

model = GRYDRA::Networks::EasyNetwork.new(print_epochs: false)
model.configure_adam_optimizer(alpha: 0.001)

model.train_numerical(historical_sales, sales, [[14, 10, 6, 1]], 0.03, 2500, :zscore,
                     lambda_l2: 0.001, patience: 180)

# Forecast future sales
future_scenarios = [
  [8, 3, 0, 88, 5500, 0, 1100],   # Regular August day
  [9, 1, 1, 78, 8500, 0, 1350],   # Labor Day
  [10, 5, 0, 65, 6000, 1, 1050]   # October with competitor promo
]

predictions = model.predict_numerical(future_scenarios, :zscore)

puts "📈 Sales Forecasting:"
future_scenarios.each_with_index do |scenario, i|
  forecast = predictions[i][0]
  month_names = %w[Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec]
  
  puts "Scenario #{i + 1}: #{month_names[scenario[0] - 1]}, Day #{scenario[1]}, Holiday=#{scenario[2] == 1 ? 'Yes' : 'No'}"
  puts "  Marketing: $#{scenario[4]}, Competitor Promo: #{scenario[5] == 1 ? 'Yes' : 'No'}"
  puts "  Forecasted Sales: $#{forecast.round(1)}k"
  puts "  Confidence: #{forecast > 60 ? 'High season' : (forecast > 45 ? 'Normal' : 'Low season')}"
  puts "  Inventory Recommendation: #{(forecast * 25).round(0)} units\n\n"
end
```

## Features

### Core Components
- **7 Activation Functions**: tanh, relu, sigmoid, leaky_relu, swish, gelu, softmax
- **8 Loss Functions**: MSE, MAE, Huber, Binary/Categorical Cross-Entropy, Hinge, Log-Cosh, Quantile
- **5 Optimizers**: Adam, SGD (with momentum/Nesterov), RMSprop, AdaGrad, AdamW
- **3 Layer Types**: Dense (fully connected), Convolutional, LSTM
- **6 Training Callbacks**: EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, ProgressBar

### Advanced Features
- **Regularization**: L1 (Lasso), L2 (Ridge), Dropout
- **Normalization**: Z-score, Min-Max, Feature-wise
- **Model Validation**: K-fold cross-validation, train/test split
- **Hyperparameter Tuning**: Grid search with parallel evaluation
- **Dimensionality Reduction**: PCA with power iteration
- **Text Processing**: Vocabulary creation, TF-IDF vectorization
- **Model Persistence**: Save/load models and vocabularies
- **Visualization**: ASCII architecture plots, gradient analysis, training curves
- **Ensemble Learning**: Multiple subnets with weighted voting

### Network Architectures
- **EasyNetwork**: High-level API for quick prototyping
- **MainNetwork**: Ensemble of multiple subnets
- **NeuralNetwork**: Low-level customizable architecture
- **Custom Layers**: Build your own layer types

## License

GPL-3.0-or-later

## Links

- GitHub: https://github.com/grcodedigitalsolutions/GRydra
- RubyGems: https://rubygems.org/gems/grydra

---

## 💙 Support Gabo-Razo

<center>If this project has been useful, consider supporting **Gabo-Razo** via GitHub Sponsors. Your contribution helps keep development active and improve future releases.</center>

---

<div align="center">
<img src="https://avatars.githubusercontent.com/u/219750358?v=4" width="90" style="border-radius: 50%; margin-bottom: 12px;"/>
<p style="font-size: 22px; font-weight: 800; margin: 0;">Gabo-Razo</p>
<p><strong>Developer • Ruby • Python • C++ • Java • Dart • COBOL</strong></p>
<p><a href="https://github.com/Gabo-Razo?tab=followers">⭐ Follow on GitHub</a></p>
<a href="https://github.com/sponsors/Gabo-Razo"><img src="https://img.shields.io/badge/Sponsor_Me-FF4081?style=for-the-badge&logo=githubsponsors&logoColor=white" height="44"/></a>
</div>

---
