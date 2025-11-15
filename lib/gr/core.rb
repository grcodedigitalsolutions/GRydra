module GRYDRA
  require 'set'
  def self.tanh(x)
    Math.tanh(x)
  end
  def self.derivative_tanh(x)
    1 - tanh(x)**2
  end
  def self.relu(x)
    x > 0 ? x : 0
  end
  def self.derivative_relu(x)
    x > 0 ? 1 : 0
  end
  def self.sigmoid(x)
    1.0 / (1.0 + Math.exp(-x))
  end
  def self.derivative_sigmoid(x)
    s = sigmoid(x)
    s * (1 - s)
  end
  def self.softmax(vector)
    max = vector.max
    exps = vector.map { |v| Math.exp(v - max) }
    sum = exps.sum
    exps.map { |v| v / sum }
  end
  def self.leaky_relu(x, alpha = 0.01)
    x > 0 ? x : alpha * x
  end
  def self.derivative_leaky_relu(x, alpha = 0.01)
    x > 0 ? 1 : alpha
  end
  def self.swish(x)
    x * sigmoid(x)
  end
  def self.derivative_swish(x)
    s = sigmoid(x)
    s + x * s * (1 - s)
  end
  def self.gelu(x)
    0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math::PI) * (x + 0.044715 * x**3)))
  end
  def self.derivative_gelu(x)
    tanh_arg = Math.sqrt(2 / Math::PI) * (x + 0.044715 * x**3)
    tanh_val = Math.tanh(tanh_arg)
    sech2 = 1 - tanh_val**2
    0.5 * (1 + tanh_val) + 0.5 * x * sech2 * Math.sqrt(2 / Math::PI) * (1 + 3 * 0.044715 * x**2)
  end
  def self.apply_dropout(outputs, dropout_rate = 0.5, training = true)
    return outputs unless training
    outputs.map { |s| rand < dropout_rate ? 0 : s / (1 - dropout_rate) }
  end
  def self.l1_regularization(weights, lambda_l1)
    lambda_l1 * weights.sum { |p| p.abs }
  end
  def self.l2_regularization(weights, lambda_l2)
    lambda_l2 * weights.sum { |p| p**2 }
  end
  def self.xavier_init(num_inputs)
    limit = Math.sqrt(6.0 / num_inputs)
    rand * 2 * limit - limit
  end
  def self.he_init(num_inputs)
    Math.sqrt(2.0 / num_inputs) * (rand * 2 - 1)
  end
  def self.zscore_normalize(data)
    n = data.size
    means = data.first.size.times.map { |i| data.map { |row| row[i] }.sum.to_f / n }
    std_devs = data.first.size.times.map do |i|
      m = means[i]
      Math.sqrt(data.map { |row| (row[i] - m)**2 }.sum.to_f / n)
    end
    normalized = data.map do |row|
      row.each_with_index.map { |value, i| std_devs[i] != 0 ? (value - means[i]) / std_devs[i] : 0 }
    end
    [normalized, means, std_devs]
  end
  def self.zscore_denormalize(normalized, means, std_devs)
    normalized.map do |row|
      row.each_with_index.map { |value, i| value * std_devs[i] + means[i] }
    end
  end
  def self.min_max_normalize(data, min_val = 0, max_val = 1)
    data_min = data.flatten.min
    data_max = data.flatten.max
    range = data_max - data_min
    return data if range == 0
    data.map do |row|
      row.map { |v| min_val + (v - data_min) * (max_val - min_val) / range }
    end
  end
  def self.mse(predictions, actuals)
    n = predictions.size
    sum = predictions.zip(actuals).map { |p, r| (p - r)**2 }.sum
    sum / n.to_f
  end
  def self.mae(predictions, actuals)
    n = predictions.size
    sum = predictions.zip(actuals).map { |p, r| (p - r).abs }.sum
    sum / n.to_f
  end
  def self.precision(tp, fp)
    tp.to_f / (tp + fp)
  end
  def self.recall(tp, fn)
    tp.to_f / (tp + fn)
  end
  def self.f1(precision, recall)
    2 * (precision * recall) / (precision + recall)
  end
  def self.confusion_matrix(predictions, actuals, threshold = 0.5)
    tp = fp = tn = fn = 0
    predictions.zip(actuals).each do |pred, actual|
      pred_bin = pred > threshold ? 1 : 0
      case [pred_bin, actual]
      when [1, 1] then tp += 1
      when [1, 0] then fp += 1
      when [0, 0] then tn += 1
      when [0, 1] then fn += 1
      end
    end
    { tp: tp, fp: fp, tn: tn, fn: fn }
  end
  def self.auc_roc(predictions, actuals)
    # Implementation of Area Under the ROC Curve
    pairs = predictions.zip(actuals).sort_by { |pred, _| -pred }
    positives = actuals.count(1)
    negatives = actuals.count(0)
    return 0.5 if positives == 0 || negatives == 0
    auc = 0.0
    fp = 0
    pairs.each do |_, actual|
      if actual == 1
        auc += fp
      else
        fp += 1
      end
    end
    auc / (positives * negatives).to_f
  end
  def self.accuracy(predictions, actuals, threshold = 0.5)
    correct = predictions.zip(actuals).count { |pred, actual| (pred > threshold ? 1 : 0) == actual }
    correct.to_f / predictions.size
  end
  ### ADVANCED OPTIMIZERS ###
  class AdamOptimizer
    def initialize(alpha = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8)
      @alpha = alpha
      @beta1 = beta1
      @beta2 = beta2
      @epsilon = epsilon
      @m = {}
      @v = {}
      @t = 0
    end
    def update(parameter_id, gradient)
      @t += 1
      @m[parameter_id] ||= 0
      @v[parameter_id] ||= 0
      @m[parameter_id] = @beta1 * @m[parameter_id] + (1 - @beta1) * gradient
      @v[parameter_id] = @beta2 * @v[parameter_id] + (1 - @beta2) * gradient**2
      m_hat = @m[parameter_id] / (1 - @beta1**@t)
      v_hat = @v[parameter_id] / (1 - @beta2**@t)
      @alpha * m_hat / (Math.sqrt(v_hat) + @epsilon)
    end
    def reset
      @m.clear
      @v.clear
      @t = 0
    end
  end
  ### CROSS-VALIDATION ###
  def self.cross_validation(data_input, data_output, k_folds = 5)
    indices = (0...data_input.size).to_a.shuffle
    fold_size = data_input.size / k_folds
    errors = []
    k_folds.times do |i|
      start = i * fold_size
      finish = [start + fold_size, data_input.size].min
      indices_test = indices[start...finish]
      indices_train = indices - indices_test
      # Split data
      train_x = indices_train.map { |idx| data_input[idx] }
      train_y = indices_train.map { |idx| data_output[idx] }
      test_x = indices_test.map { |idx| data_input[idx] }
      test_y = indices_test.map { |idx| data_output[idx] }
      # Train and evaluate
      error = yield(train_x, train_y, test_x, test_y)
      errors << error
    end
    {
      errors: errors,
      average: errors.sum / errors.size.to_f,
      deviation: Math.sqrt(errors.map { |e| (e - errors.sum / errors.size.to_f)**2 }.sum / errors.size.to_f)
    }
  end
  ### ANALYSIS AND VISUALIZATION ###
  def self.analyze_gradients(model)
    gradients = []
    if model.respond_to?(:layers)
      model.layers.each do |layer|
        layer.neurons.each do |neuron|
          gradients << neuron.delta.abs if neuron.delta
        end
      end
    elsif model.respond_to?(:subnets)
      model.subnets.each do |subnet|
        subnet.layers.each do |layer|
          layer.neurons.each do |neuron|
            gradients << neuron.delta.abs if neuron.delta
          end
        end
      end
    end
    return { message: 'No gradients to analyze' } if gradients.empty?
    average = gradients.sum / gradients.size.to_f
    {
      average: average,
      maximum: gradients.max,
      minimum: gradients.min,
      deviation: Math.sqrt(gradients.map { |g| (g - average)**2 }.sum / gradients.size.to_f),
      total_parameters: gradients.size
    }
  end
  def self.plot_architecture_ascii(model)
    puts "
Network Architecture:"
    puts '=' * 50
    if model.respond_to?(:subnets)
      model.subnets.each_with_index do |subnet, idx|
        puts "
 Subnet #{idx + 1}:"
        plot_individual_network(subnet)
      end
    else
      plot_individual_network(model)
    end
    puts '=' * 50
  end
  def self.plot_individual_network(network)
    network.layers.each_with_index do |layer, i|
      neurons = layer.neurons.size
      activation = layer.activation || :linear
      # Visual representation of neurons
      symbols = if neurons <= 10
                   '●' * neurons
                 else
                   '●' * 8 + "... (#{neurons})"
                 end
      puts "  Layer #{i + 1}: #{symbols} [#{activation}]"
      puts "           #{' ' * 8}↓" unless i == network.layers.size - 1
    end
  end
  ### ADVANCED PREPROCESSING ###
  def self.pca(data, components = 2)
    n = data.size
    m = data.first.size
    # Center data
    means = (0...m).map { |i| data.map { |row| row[i] }.sum.to_f / n }
    centered_data = data.map { |row| row.zip(means).map { |v, mean| v - mean } }
    # Calculate covariance matrix (simplified)
    covariance = Array.new(m) { Array.new(m, 0) }
    (0...m).each do |i|
      (0...m).each do |j|
        covariance[i][j] = centered_data.map { |row| row[i] * row[j] }.sum / (n - 1).to_f
      end
    end
    # Note: A complete PCA implementation would require eigenvalue/eigenvector calculation
    # This is a simplified version that returns the first components
    puts '  Simplified PCA - For complete analysis use specialized libraries'
    {
      means: means,
      covariance: covariance,
      centered_data: centered_data[0...components]
    }
  end
  ### UTILITY FUNCTIONS ###
  def self.split_data(data_x, data_y, training_ratio = 0.8, seed = nil)
    srand(seed) if seed
    indices = (0...data_x.size).to_a.shuffle
    cut = (data_x.size * training_ratio).to_i
    {
      train_x: indices[0...cut].map { |i| data_x[i] },
      train_y: indices[0...cut].map { |i| data_y[i] },
      test_x: indices[cut..-1].map { |i| data_x[i] },
      test_y: indices[cut..-1].map { |i| data_y[i] }
    }
  end
  def self.hyperparameter_search(data_x, data_y, param_grid)
    best_params = nil
    best_score = Float::INFINITY
    results = []
    puts ' Starting hyperparameter search...'
    param_grid.each_with_index do |params, idx|
      puts "  Testing configuration #{idx + 1}/#{param_grid.size}: #{params}"
      begin
        score = yield(params, data_x, data_y)
        results << { parameters: params, score: score }
        if score < best_score
          best_score = score
          best_params = params
          puts "     New best configuration found! Score: #{score.round(6)}"
        else
          puts "     Score: #{score.round(6)}"
        end
      rescue StandardError => e
        puts "     Error with this configuration: #{e.message}"
        results << { parameters: params, score: Float::INFINITY, error: e.message }
      end
    end
    puts "
 Best parameters found:"
    puts "   Configuration: #{best_params}"
    puts "   Score: #{best_score.round(6)}"
    {
      parameters: best_params,
      score: best_score,
      all_results: results.sort_by { |r| r[:score] }
    }
  end
  def self.generate_synthetic_data(n_samples, n_features, noise = 0.1, seed = nil)
    srand(seed) if seed
    data = Array.new(n_samples) do
      Array.new(n_features) { rand * 2 - 1 + (rand * noise - noise / 2) }
    end
    # Generate labels based on a simple function
    labels = data.map do |sample|
      # Simple function: weighted sum + noise
      value = sample.each_with_index.sum { |x, i| x * (i + 1) * 0.1 }
      [value + (rand * noise - noise / 2)]
    end
    { data: data, labels: labels }
  end
  ### NEURON CLASS ###
  class Neuron
    attr_accessor :weights, :bias, :output, :delta
    def initialize(inputs, activation = :tanh)
      unless inputs.is_a?(Integer) && inputs > 0
        raise ArgumentError,
              'Number of inputs must be a positive integer'
      end
      @weights = case activation
               when :relu, :leaky_relu
                 Array.new(inputs) { GRYDRA.he_init(inputs) }
               else
                 Array.new(inputs) { GRYDRA.xavier_init(inputs) }
               end
      @bias = case activation
               when :relu, :leaky_relu
                 GRYDRA.he_init(inputs)
               else
                 GRYDRA.xavier_init(inputs)
               end
      @output = 0
      @delta = 0
      @activation = activation
      @sum = 0
      @dropout_mask = nil
    end
    def calculate_output(inputs, apply_dropout = false, dropout_rate = 0.5)
      unless inputs.is_a?(Array) && inputs.all? { |e| e.is_a?(Numeric) }
        raise ArgumentError,
              'Inputs must be an array of numbers'
      end
      raise "Error: inputs (#{inputs.size}) do not match weights (#{@weights.size})" if @weights.size != inputs.size
      @sum = @weights.zip(inputs).map { |weight, input| weight * input }.sum + @bias
      @output = case @activation
                when :tanh then GRYDRA.tanh(@sum)
                when :relu then GRYDRA.relu(@sum)
                when :sigmoid then GRYDRA.sigmoid(@sum)
                when :leaky_relu then GRYDRA.leaky_relu(@sum)
                when :swish then GRYDRA.swish(@sum)
                when :gelu then GRYDRA.gelu(@sum)
                else @sum
                end
      if apply_dropout
        @dropout_mask = rand < dropout_rate ? 0 : 1 / (1 - dropout_rate)
        @output *= @dropout_mask
      end
      @output
    end
    def derivative_activation
      case @activation
      when :tanh then GRYDRA.derivative_tanh(@output)
      when :relu then GRYDRA.derivative_relu(@output)
      when :sigmoid then GRYDRA.derivative_sigmoid(@sum)
      when :leaky_relu then GRYDRA.derivative_leaky_relu(@sum)
      when :swish then GRYDRA.derivative_swish(@sum)
      when :gelu then GRYDRA.derivative_gelu(@sum)
      else 1
      end
    end
  end
  ### BASE LAYER CLASS ###
  class Layer
    def calculate_outputs(inputs)
      raise NotImplementedError, 'Implement in subclass'
    end
  end
  ### DENSE LAYER CLASS ###
  class DenseLayer < Layer
    attr_accessor :neurons, :activation
    def initialize(num_neurons, inputs_per_neuron, activation = :tanh)
      @activation = activation
      @neurons = Array.new(num_neurons) { Neuron.new(inputs_per_neuron, activation) }
    end
    def calculate_outputs(inputs, apply_dropout = false, dropout_rate = 0.5)
      @neurons.map { |neuron| neuron.calculate_output(inputs, apply_dropout, dropout_rate) }
    end
  end
  class ConvLayer < Layer
    attr_accessor :filters, :kernel_size, :stride, :padding
    def initialize(filters, kernel_size, stride = 1, padding = 0, activation = :relu)
      @filters = filters
      @kernel_size = kernel_size
      @stride = stride
      @padding = padding
      @activation = activation
      @weights = Array.new(filters) { Array.new(kernel_size * kernel_size) { rand * 0.1 - 0.05 } }
      @biases = Array.new(filters) { rand * 0.1 - 0.05 }
    end
    def calculate_outputs(input)
      # Simplified 2D convolution implementation
      # For a complete implementation, 2D matrix handling would be needed
      puts '⚠️  Simplified Convolutional Layer - For full use, implement 2D convolution'
      input.map { |x| x * 0.5 } # Placeholder
    end
  end
  class LSTMLayer < Layer
    attr_accessor :units
    def initialize(units, inputs_per_unit)
      @units = units
      @inputs_per_unit = inputs_per_unit
      # Initialize gates (forget, input, output)
      @forget_gate = DenseLayer.new(units, inputs_per_unit + units, :sigmoid)
      @input_gate = DenseLayer.new(units, inputs_per_unit + units, :sigmoid)
      @output_gate = DenseLayer.new(units, inputs_per_unit + units, :sigmoid)
      @candidates = DenseLayer.new(units, inputs_per_unit + units, :tanh)
      @cell_state = Array.new(units, 0)
      @hidden_state = Array.new(units, 0)
    end
    def calculate_outputs(input)
      # Simplified LSTM implementation
      puts '⚠️  Simplified LSTM Layer - For full use, implement all gates'
      combined_input = input + @hidden_state
      # Calculate gates
      f_t = @forget_gate.calculate_outputs(combined_input)
      i_t = @input_gate.calculate_outputs(combined_input)
      o_t = @output_gate.calculate_outputs(combined_input)
      c_candidate = @candidates.calculate_outputs(combined_input)
      # Update cell state
      @cell_state = @cell_state.zip(f_t, i_t, c_candidate).map do |c, f, i, candidate|
        f * c + i * candidate
      end
      # Calculate output
      @hidden_state = o_t.zip(@cell_state).map { |o, c| o * Math.tanh(c) }
      @hidden_state
    end
  end
  ### NEURAL NETWORK CLASS ###
  class NeuralNetwork
    attr_accessor :layers
    def initialize(structure, print_epochs = false, plot = false, activations = nil)
      @print_epochs = print_epochs
      @plot = plot
      @layers = []
      @history_error = []
      @optimizer = nil
      activations ||= Array.new(structure.size - 1, :tanh)
      structure.each_cons(2).with_index do |(inputs, outputs), i|
        @layers << DenseLayer.new(outputs, inputs, activations[i])
      end
    end
    def use_adam_optimizer(alpha = 0.001, beta1 = 0.9, beta2 = 0.999)
      @optimizer = AdamOptimizer.new(alpha, beta1, beta2)
    end
    def calculate_outputs(inputs, apply_dropout = false, dropout_rate = 0.5)
      raise ArgumentError, 'Inputs must be an array of numbers' unless inputs.is_a?(Array) && inputs.all? { |e| e.is_a?(Numeric) }
      @layers.inject(inputs) { |outputs, layer| layer.calculate_outputs(outputs, apply_dropout, dropout_rate) }
    end
    # Training with mini-batch, early stopping, decay learning rate, and regularization
    def train(data_input, data_output, learning_rate, epochs, error_threshold = nil,
                 batch_size: 1, patience: 50, decay: 0.95, lambda_l1: 0, lambda_l2: 0,
                 dropout: false, dropout_rate: 0.5)
      best_error = Float::INFINITY
      patience_counter = 0
      epochs.times do |epoch|
        error_total = 0
        error_regularization = 0
        # Shuffle data
        indices = (0...data_input.size).to_a.shuffle
        data_input = indices.map { |i| data_input[i] }
        data_output = indices.map { |i| data_output[i] }
        data_input.each_slice(batch_size).with_index do |batch_inputs, batch_idx|
          batch_outputs_real = data_output[batch_idx * batch_size, batch_size]
          batch_inputs.zip(batch_outputs_real).each do |input, output_real|
            outputs = calculate_outputs(input, dropout, dropout_rate)
            errors = outputs.zip(output_real).map { |output, real| real - output }
            error_total += errors.map { |e| e**2 }.sum / errors.size
            if lambda_l1 > 0 || lambda_l2 > 0
              @layers.each do |layer|
                layer.neurons.each do |neuron|
                  error_regularization += GRYDRA.l1_regularization(neuron.weights, lambda_l1) if lambda_l1 > 0
                  error_regularization += GRYDRA.l2_regularization(neuron.weights, lambda_l2) if lambda_l2 > 0
                end
              end
            end
            # Output layer
            @layers.last.neurons.each_with_index do |neuron, i|
              neuron.delta = errors[i] * neuron.derivative_activation
            end
            # Backpropagation hidden layers
            (@layers.size - 2).downto(0) do |i|
              @layers[i].neurons.each_with_index do |neuron, j|
                sum_deltas = @layers[i + 1].neurons.sum { |n| n.weights[j] * n.delta }
                neuron.delta = sum_deltas * neuron.derivative_activation
              end
            end
            # Update weights and bias
            @layers.each_with_index do |layer, idx|
              inputs_layer = idx.zero? ? input : @layers[idx - 1].neurons.map(&:output)
              layer.neurons.each_with_index do |neuron, neuron_idx|
                neuron.weights.each_with_index do |_weight, i|
                  gradient = neuron.delta * inputs_layer[i]
                  if @optimizer
                    param_id = "layer_#{idx}_neuron_#{neuron_idx}_weight_#{i}"
                    update = @optimizer.update(param_id, gradient)
                    neuron.weights[i] += update
                  else
                    neuron.weights[i] += learning_rate * gradient
                  end
                  # Apply regularization to weights
                  if lambda_l1 > 0
                    neuron.weights[i] -= learning_rate * lambda_l1 * (neuron.weights[i] > 0 ? 1 : -1)
                  end
                  neuron.weights[i] -= learning_rate * lambda_l2 * 2 * neuron.weights[i] if lambda_l2 > 0
                end
                # Update bias
                if @optimizer
                  param_id = "layer_#{idx}_neuron_#{neuron_idx}_bias"
                  update = @optimizer.update(param_id, neuron.delta)
                  neuron.bias += update
                else
                  neuron.bias += learning_rate * neuron.delta
                end
              end
            end
          end
        end
        error_total += error_regularization
        if error_threshold && error_total < error_threshold
          puts "Error threshold reached at epoch #{epoch + 1}: #{error_total}"
          break
        end
        if error_total < best_error
          best_error = error_total
          patience_counter = 0
        else
          patience_counter += 1
          if patience_counter >= patience
            puts "Early stopping at epoch #{epoch + 1}, error has not improved for #{patience} epochs."
            break
          end
        end
        learning_rate *= decay
        @history_error << error_total if @plot
        if @print_epochs
          puts "Epoch #{epoch + 1}, Total Error: #{error_total.round(6)}, learning rate: #{learning_rate.round(6)}"
        end
      end
      GRYDRA.plot_error(@history_error) if @plot
    end
    def info_network
      puts "Neural network with #{@layers.size} layers:"
      @layers.each_with_index do |layer, i|
        puts " Layer #{i + 1}: #{layer.neurons.size} neurons, activation: #{layer.activation}"
        layer.neurons.each_with_index do |neuron, j|
          puts "  Neuron #{j + 1}: Weights=#{neuron.weights.map { |p| p.round(3) }}, Bias=#{neuron.bias.round(3)}"
        end
      end
    end
    # Export to DOT for Graphviz
    def export_graphviz(filename = 'neural_network.dot')
      File.open(filename, 'w') do |f|
        f.puts 'digraph NeuralNetwork {'
        @layers.each_with_index do |layer, i|
          layer.neurons.each_with_index do |_neuron, j|
            node = "L#{i}_N#{j}"
            f.puts "  #{node} [label=\"N#{j + 1}\"];"
            next unless i < @layers.size - 1
            @layers[i + 1].neurons.each_with_index do |next_neuron, k|
              weight = next_neuron.weights[j].round(3)
              f.puts "  #{node} -> L#{i + 1}_N#{k} [label=\"#{weight}\"];"
            end
          end
        end
        f.puts '}'
      end
      puts "Network exported to #{filename} (Graphviz DOT)"
    end
  end
  ### MAIN NETWORK ###
  class MainNetwork
    attr_accessor :subnets
    def initialize(print_epochs = false, plot = false)
      @subnets = []
      @print_epochs = print_epochs
      @plot = plot
    end
    def add_subnet(structure, activations = nil)
      @subnets << NeuralNetwork.new(structure, @print_epochs, @plot, activations)
    end
    def train_subnets(data, learning_rate, epochs, **opts)
      data.each_with_index do |data_subnet, index|
        puts "Training Subnet #{index + 1}..."
        @subnets[index].train(data_subnet[:input], data_subnet[:output], learning_rate, epochs, **opts)
      end
    end
    def combine_results(input_main)
      outputs_subnets = @subnets.map { |subnet| subnet.calculate_outputs(input_main) }
      outputs_subnets.transpose.map { |outputs| outputs.sum / outputs.size }
    end
    def combine_results_weighted(input_main, weights = nil)
      outputs_subnets = @subnets.map { |subnet| subnet.calculate_outputs(input_main) }
      weights ||= Array.new(@subnets.size, 1.0 / @subnets.size)
      outputs_subnets.transpose.map do |outputs|
        outputs.zip(weights).map { |output, weight| output * weight }.sum
      end
    end
  end
  ### SAVE AND LOAD MODEL AND VOCABULARY ###
  def self.save_model(model, name, path = Dir.pwd, vocabulary = nil)
    file_path = File.join(path, "#{name}.net")
    # Open file in binary write mode and save the serialized object
    File.open(file_path, 'wb') { |f| Marshal.dump(model, f) }
    puts "\e[33mModel saved to '#{file_path}'\e[0m"
    # If vocabulary is passed, delegate saving it to another function
    return unless vocabulary
    save_vocabulary(vocabulary, name, path)
  end
  def self.load_model(name, path = Dir.pwd)
    model = nil
    file_path = File.join(path, "#{name}.net")
    File.open(file_path, 'rb') { |f| model = Marshal.load(f) }
    model
  end
  def self.save_vocabulary(vocabulary, name, path = Dir.pwd)
    file_path = File.join(path, "#{name}_vocab.bin")
    File.open(file_path, 'wb') { |f| Marshal.dump(vocabulary, f) }
    puts "\e[33mVocabulary saved to '#{file_path}'\e[0m"
  end
  def self.load_vocabulary(name, path = Dir.pwd)
    vocabulary = nil
    file_path = File.join(path, "#{name}_vocab.bin")
    File.open(file_path, 'rb') { |f| vocabulary = Marshal.load(f) }
    vocabulary
  end
  ### UPDATED PREPROCESSING ###
  def self.normalize_multiple(data, max_values, method = :max)
    case method
    when :max
      data.map do |row|
        row.each_with_index.map { |value, idx| value.to_f / max_values[idx] }
      end
    when :zscore
      means = max_values[:means]
      std_devs = max_values[:std_devs]
      data.map do |row|
        row.each_with_index.map do |value, idx|
          std_devs[idx] != 0 ? (value.to_f - means[idx]) / std_devs[idx] : 0
        end
      end
    else
      raise 'Unknown normalization method'
    end
  end
  def self.calculate_max_values(data, method = :max)
    if method == :max
      max_values = {}
      data.first.size.times do |i|
        max_values[i] = data.map { |row| row[i] }.max.to_f
      end
      max_values
    elsif method == :zscore
      n = data.size
      means = data.first.size.times.map do |i|
        data.map { |row| row[i] }.sum.to_f / n
      end
      std_devs = data.first.size.times.map do |i|
        m = means[i]
        Math.sqrt(data.map { |row| (row[i] - m)**2 }.sum.to_f / n)
      end
      { means: means, std_devs: std_devs }
    else
      raise 'Unknown method for calculating max values'
    end
  end
  ### TEXT FUNCTIONS ###
  def self.create_vocabulary(texts)
    texts.map(&:split).flatten.map(&:downcase).uniq
  end
  def self.vectorize_text(text, vocabulary)
    vector = Array.new(vocabulary.size, 0)
    words = text.downcase.split
    words.each do |word|
      index = vocabulary.index(word)
      vector[index] = 1 if index
    end
    vector
  end
  def self.normalize_with_vocabulary(data, vocabulary)
    max_value = vocabulary.size
    data.map { |vector| vector.map { |v| v.to_f / max_value } }
  end
  def self.create_advanced_vocabulary(texts, min_frequency = 1, max_words = nil)
    # Count frequencies
    frequencies = Hash.new(0)
    texts.each do |text|
      text.downcase.split.each { |word| frequencies[word] += 1 }
    end
    # Filter by minimum frequency
    vocabulary = frequencies.select { |_, freq| freq >= min_frequency }.keys
    # Limit size if specified
    if max_words && vocabulary.size > max_words
      vocabulary = frequencies.sort_by { |_, freq| -freq }.first(max_words).map(&:first)
    end
    vocabulary.sort
  end
  def self.vectorize_text_tfidf(text, vocabulary, corpus_frequencies)
    vector = Array.new(vocabulary.size, 0.0)
    words = text.downcase.split
    doc_frequencies = Hash.new(0)
    # Count document frequencies
    words.each { |word| doc_frequencies[word] += 1 }
    # Calculate TF-IDF
    vocabulary.each_with_index do |word, idx|
      next unless doc_frequencies[word] > 0
      tf = doc_frequencies[word].to_f / words.size
      idf = Math.log(corpus_frequencies.size.to_f / (corpus_frequencies[word] || 1))
      vector[idx] = tf * idf
    end
    vector
  end
  ### CLASS EasyNetwork (unchanged, just adding zscore normalization and activations option) ###
  class EasyNetwork
    attr_accessor :network, :vocabulary, :max_values, :max_values_output
    def initialize(print_epochs = false, plot = false)
      @network = GRYDRA::MainNetwork.new(print_epochs, plot)
      @vocabulary = nil
      @max_values = {}
      @max_values_output = {}
    end
    def configure_adam_optimizer(alpha = 0.001, beta1 = 0.9, beta2 = 0.999)
      @network.subnets.each { |subnet| subnet.use_adam_optimizer(alpha, beta1, beta2) }
    end
    def evaluate_model(data_test_x, data_test_y, metrics = %i[mse mae])
      predictions = predict_numerical(data_test_x)
      results = {}
      predictions_flat = predictions.flatten
      actuals_flat = data_test_y.flatten
      metrics.each do |metric|
        case metric
        when :mse
          results[:mse] = GRYDRA.mse(predictions_flat, actuals_flat)
        when :mae
          results[:mae] = GRYDRA.mae(predictions_flat, actuals_flat)
        when :accuracy
          results[:accuracy] = GRYDRA.accuracy(predictions_flat, actuals_flat)
        when :confusion_matrix
          results[:confusion_matrix] = GRYDRA.confusion_matrix(predictions_flat, actuals_flat)
        end
      end
      results
    end
    # --------- For hash-type data ---------
    def train_hashes(data_hash, input_keys, label_key, structures, rate, epochs, normalization = :max,
                        **opts)
      @network.subnets.clear # clear previous subnets
      inputs = data_hash.map do |item|
        input_keys.map do |key|
          value = item[key]
          if value == true
            1.0
          else
            value == false ? 0.0 : value.to_f
          end
        end
      end
      @max_values = GRYDRA.calculate_max_values(inputs, normalization)
      data_normalized = GRYDRA.normalize_multiple(inputs, @max_values, normalization)
      labels = data_hash.map { |item| [item[label_key].to_f] }
      @max_values_output = GRYDRA.calculate_max_values(labels, normalization)
      labels_no = GRYDRA.normalize_multiple(labels, @max_values_output, normalization)
      structures.each do |structure|
        @network.add_subnet([input_keys.size, *structure])
      end
      data_for_subnets = structures.map do |_|
        { input: data_normalized, output: labels_no }
      end
      @network.train_subnets(data_for_subnets, rate, epochs, **opts)
    end
    def predict_hashes(new_hashes, input_keys, normalization = :max)
      inputs = new_hashes.map do |item|
        input_keys.map do |key|
          value = item[key]
          if value == true
            1.0
          else
            value == false ? 0.0 : value.to_f
          end
        end
      end
      data_normalized = GRYDRA.normalize_multiple(inputs, @max_values, normalization)
      data_normalized.map do |input|
        pred_norm = @network.combine_results(input)
        if normalization == :zscore && @max_values_output.is_a?(Hash)
          pred_norm.map.with_index do |val, idx|
            val * @max_values_output[:std_devs][idx] + @max_values_output[:means][idx]
          end
        else
          pred_norm.map.with_index { |val, idx| val * @max_values_output[idx] }
        end
      end
    end
    # --------- For numerical data ---------
    def train_numerical(data_input, data_output, structures, rate, epochs, normalization = :max, **opts)
      @network.subnets.clear # clear previous subnets
      @max_values = GRYDRA.calculate_max_values(data_input, normalization)
      @max_values_output = GRYDRA.calculate_max_values(data_output, normalization)
      data_input_no = GRYDRA.normalize_multiple(data_input, @max_values, normalization)
      data_output_no = GRYDRA.normalize_multiple(data_output, @max_values_output, normalization)
      structures.each do |structure|
        @network.add_subnet([data_input.first.size, *structure])
      end
      data_for_subnets = structures.map do |_|
        { input: data_input_no, output: data_output_no }
      end
      @network.train_subnets(data_for_subnets, rate, epochs, **opts)
    end
    def predict_numerical(new_data, normalization = :max)
      data_normalized = GRYDRA.normalize_multiple(new_data, @max_values, normalization)
      data_normalized.map do |input|
        pred_norm = @network.combine_results(input)
        if normalization == :zscore && @max_values_output.is_a?(Hash)
          pred_norm.map.with_index do |val, idx|
            val * @max_values_output[:std_devs][idx] + @max_values_output[:means][idx]
          end
        else
          pred_norm.map.with_index { |val, idx| val * @max_values_output[idx] }
        end
      end
    end
    # --------- For text data ---------
    def train_text(texts, labels, structures, rate, epochs, normalization = :max, **opts)
      @network.subnets.clear # clear previous subnets
      @vocabulary = GRYDRA.create_vocabulary(texts)
      inputs = texts.map { |text| GRYDRA.vectorize_text(text, @vocabulary) }
      @max_values = { 0 => @vocabulary.size } # Only vocabulary size for text
      data_normalized = GRYDRA.normalize_multiple(inputs, @max_values, normalization)
      @max_values_output = GRYDRA.calculate_max_values(labels, normalization)
      labels_no = GRYDRA.normalize_multiple(labels, @max_values_output, normalization)
      structures.each do |structure|
        @network.add_subnet([@vocabulary.size, *structure])
      end
      data_for_subnets = structures.map do |_|
        { input: data_normalized, output: labels_no }
      end
      @network.train_subnets(data_for_subnets, rate, epochs, **opts)
    end
    def predict_text(new_texts, normalization = :max)
      inputs = new_texts.map { |text| GRYDRA.vectorize_text(text, @vocabulary) }
      data_normalized = GRYDRA.normalize_multiple(inputs, @max_values, normalization)
      data_normalized.map do |input|
        pred_norm = @network.combine_results(input)
        if normalization == :zscore && @max_values_output.is_a?(Hash)
          pred_norm.map.with_index do |val, idx|
            val * @max_values_output[:std_devs][idx] + @max_values_output[:means][idx]
          end
        else
          pred_norm.map.with_index { |val, idx| val * @max_values_output[idx] }
        end
      end
    end
  end
  METHOD_DESCRIPTIONS = {
    # MainNetwork
    'MainNetwork.add_subnet' => {
      description: 'Adds a subnet to the main network with the given structure. The structure defines the number of neurons per layer (including inputs).',
      example: <<~EX
        network = GRYDRA::MainNetwork.new
        network.add_subnet([2, 4, 1]) # 2 inputs, 4 hidden neurons, 1 output
      EX
    },
    'MainNetwork.train_subnets' => {
      description: 'Trains all subnets using input and output data, with learning rate, epochs, and options like patience for early stopping.',
      example: <<~EX
        data = [
          {input: [[0.1, 0.2]], output: [[0.3]]},
          {input: [[0.5, 0.6]], output: [[0.7]]}
        ]
        network = GRYDRA::MainNetwork.new(true)
        network.add_subnet([2, 3, 1])
        network.add_subnet([2, 2, 1])
        network.train_subnets(data, 0.01, 1000, patience: 5, lambda_l1: 0.001, dropout: true)
      EX
    },
    'MainNetwork.combine_results' => {
      description: 'Averages the outputs of all subnets for a given input, generating the final prediction.',
      example: <<~EX
        result = network.combine_results([0.2, 0.8])
      EX
    },
    'MainNetwork.combine_results_weighted' => {
      description: 'Combines the outputs of all subnets using specific weights for each subnet.',
      example: <<~EX
        result = network.combine_results_weighted([0.2, 0.8], [0.6, 0.4])
      EX
    },
    # EasyNetwork (easier interface)
    'EasyNetwork.train_numerical' => {
      description: 'Trains the network with numerical data (arrays of numbers) for input and output. Normalizes, creates subnets, and trains.',
      example: <<~EX
        data_input = [[170, 25], [160, 30], [180, 22]]
        data_output = [[65], [60], [75]]
        structures = [[4, 1], [3, 1]]
        network = GRYDRA::EasyNetwork.new(true)
        network.train_numerical(data_input, data_output, structures, 0.05, 15000, :max,
                              lambda_l2: 0.001, dropout: true, dropout_rate: 0.3)
      EX
    },
    'EasyNetwork.predict_numerical' => {
      description: 'Predicts values with new numerical data normalized the same way as training.',
      example: <<~EX
        new_data = [[172, 26]]
        predictions = network.predict_numerical(new_data, :max)
      EX
    },
    'EasyNetwork.configure_adam_optimizer' => {
      description: 'Configures the Adam optimizer for all subnets with customizable parameters.',
      example: <<~EX
        network.configure_adam_optimizer(0.001, 0.9, 0.999)
      EX
    },
    'EasyNetwork.evaluate_model' => {
      description: 'Evaluates the model with test data using multiple metrics.',
      example: <<~EX
        results = network.evaluate_model(test_x, test_y, [:mse, :mae, :accuracy])
      EX
    },
    'EasyNetwork.train_hashes' => {
      description: 'Trains the network with input data in hash format, specifying the keys to use and the label key.',
      example: <<~EX
        data_hash = [
          {height: 170, age: 25, weight: 65},
          {height: 160, age: 30, weight: 60}
        ]
        network = GRYDRA::EasyNetwork.new(true)
        network.train_hashes(data_hash, [:height, :age], :weight, [[4, 1]], 0.05, 15000, :max)
      EX
    },
    'EasyNetwork.predict_hashes' => {
      description: 'Predicts using hash data with the specified keys for input.',
      example: <<~EX
        new_hashes = [{height: 172, age: 26}]
        predictions = network.predict_hashes(new_hashes, [:height, :age], :max)
      EX
    },
    'EasyNetwork.train_text' => {
      description: 'Trains the network with texts and numerical labels, creating a vocabulary to vectorize texts.',
      example: <<~EX
        texts = ["hello world", "good day"]
        labels = [[1], [0]]
        structures = [[5, 1]]
        network = GRYDRA::EasyNetwork.new(true)
        network.train_text(texts, labels, structures, 0.01, 5000)
      EX
    },
    'EasyNetwork.predict_text' => {
      description: 'Predicts with new texts, vectorizing according to the learned vocabulary.',
      example: <<~EX
        new_texts = ["hello"]
        predictions = network.predict_text(new_texts)
      EX
    },
    # New activation functions
    'GRYDRA.leaky_relu' => {
      description: 'Leaky ReLU activation function that allows a small gradient for negative values.',
      example: <<~EX
        result = GRYDRA.leaky_relu(-0.5, 0.01) # -0.005
      EX
    },
    'GRYDRA.swish' => {
      description: 'Swish activation function (x * sigmoid(x)) which is smooth and non-monotonic.',
      example: <<~EX
        result = GRYDRA.swish(1.0)
      EX
    },
    'GRYDRA.gelu' => {
      description: 'GELU (Gaussian Error Linear Unit) activation function used in transformers.',
      example: <<~EX
        result = GRYDRA.gelu(0.5)
      EX
    },
    # Regularization
    'GRYDRA.apply_dropout' => {
      description: 'Applies dropout to outputs during training to prevent overfitting.',
      example: <<~EX
        dropout_outputs = GRYDRA.apply_dropout([0.5, 0.8, 0.3], 0.5, true)
      EX
    },
    'GRYDRA.l1_regularization' => {
      description: 'Calculates L1 penalty (sum of absolute values) for regularization.',
      example: <<~EX
        penalty = GRYDRA.l1_regularization([0.5, -0.3, 0.8], 0.01)
      EX
    },
    'GRYDRA.l2_regularization' => {
      description: 'Calculates L2 penalty (sum of squares) for regularization.',
      example: <<~EX
        penalty = GRYDRA.l2_regularization([0.5, -0.3, 0.8], 0.01)
      EX
    },
    # Advanced metrics
    'GRYDRA.confusion_matrix' => {
      description: 'Calculates the confusion matrix for binary classification problems.',
      example: <<~EX
        matrix = GRYDRA.confusion_matrix([0.8, 0.3, 0.9], [1, 0, 1], 0.5)
      EX
    },
    'GRYDRA.auc_roc' => {
      description: 'Calculates the area under the ROC curve for classifier evaluation.',
      example: <<~EX
        auc = GRYDRA.auc_roc([0.8, 0.3, 0.9, 0.1], [1, 0, 1, 0])
      EX
    },
    'GRYDRA.accuracy' => {
      description: 'Calculates the model\'s accuracy.',
      example: <<~EX
        acc = GRYDRA.accuracy([0.8, 0.3, 0.9], [1, 0, 1], 0.5)
      EX
    },
    # Cross-validation
    'GRYDRA.cross_validation' => {
      description: 'Performs k-fold cross-validation to robustly evaluate the model.',
      example: <<~EX
        result = GRYDRA.cross_validation(data_x, data_y, 5) do |train_x, train_y, test_x, test_y|
          # train and evaluate model
          error
        end
      EX
    },
    # Analysis and visualization
    'GRYDRA.analyze_gradients' => {
      description: 'Analyzes the model\'s gradients to detect vanishing/exploding gradient problems.',
      example: <<~EX
        analysis = GRYDRA.analyze_gradients(model)
      EX
    },
    'GRYDRA.plot_architecture_ascii' => {
      description: 'Displays an ASCII representation of the network architecture.',
      example: <<~EX
        GRYDRA.plot_architecture_ascii(model)
      EX
    },
    # Utilities
    'GRYDRA.split_data' => {
      description: 'Splits data into training and test sets randomly.',
      example: <<~EX
        split = GRYDRA.split_data(data_x, data_y, 0.8, 42)
      EX
    },
    'GRYDRA.hyperparameter_search' => {
      description: 'Performs hyperparameter search using grid search.',
      example: <<~EX
        grid = [{rate: 0.01, epochs: 1000}, {rate: 0.1, epochs: 500}]
        result = GRYDRA.hyperparameter_search(data_x, data_y, grid) do |params, x, y|
          # train with params and return error
        end
      EX
    },
    'GRYDRA.generate_synthetic_data' => {
      description: 'Generates synthetic data for testing and experimentation.',
      example: <<~EX
        data = GRYDRA.generate_synthetic_data(100, 3, 0.1, 42)
      EX
    },
    'GRYDRA.min_max_normalize' => {
      description: 'Normalizes data using Min-Max scaling to a specific range.',
      example: <<~EX
        data_norm = GRYDRA.min_max_normalize(data, 0, 1)
      EX
    },
    'GRYDRA.pca' => {
      description: 'Performs Principal Component Analysis (simplified version).',
      example: <<~EX
        result = GRYDRA.pca(data, 2)
      EX
    },
    # Advanced text processing
    'GRYDRA.create_advanced_vocabulary' => {
      description: 'Creates vocabulary with frequency filtering and size limit.',
      example: <<~EX
        vocab = GRYDRA.create_advanced_vocabulary(texts, 2, 1000)
      EX
    },
    'GRYDRA.vectorize_text_tfidf' => {
      description: 'Vectorizes text using TF-IDF instead of binary vectorization.',
      example: <<~EX
        vector = GRYDRA.vectorize_text_tfidf(text, vocabulary, corpus_freqs)
      EX
    },
    # Existing methods
    'GRYDRA.describe_method' => {
      description: 'Displays example of a class or method instance.',
      example: <<~EX
        GRYDRA.describe_method("GRYDRA", "save_model")
      EX
    },
    'GRYDRA.save_model' => {
      description: 'Saves the trained model to a binary file so it can be loaded later. Optionally saves the vocabulary as well.',
      example: <<~EX
        GRYDRA.save_model(model, "my_model", "./models", vocabulary)
      EX
    },
    'GRYDRA.load_model' => {
      description: 'Loads a saved model from a binary file to use it without retraining.',
      example: <<~EX
        model = GRYDRA.load_model("my_model", "./models")
      EX
    },
    'GRYDRA.save_vocabulary' => {
      description: 'Saves the vocabulary to a binary file for later loading.',
      example: <<~EX
        GRYDRA.save_vocabulary(vocabulary, "my_model", "./models")
      EX
    },
    'GRYDRA.load_vocabulary' => {
      description: 'Loads the vocabulary from a saved binary file.',
      example: <<~EX
        vocabulary = GRYDRA.load_vocabulary("my_model", "./models")
      EX
    },
    'GRYDRA.normalize_multiple' => {
      description: 'Normalizes a set of data according to the specified method (:max or :zscore).',
      example: <<~EX
        max_values = GRYDRA.calculate_max_values(data, :max)
        data_norm = GRYDRA.normalize_multiple(data, max_values, :max)
      EX
    },
    'GRYDRA.calculate_max_values' => {
      description: 'Calculates maximum values or means and standard deviations according to the method for normalizing data.',
      example: <<~EX
        max_values = GRYDRA.calculate_max_values(data, :max)
        statistics = GRYDRA.calculate_max_values(data, :zscore)
      EX
    },
    'GRYDRA.create_vocabulary' => {
      description: 'Creates a unique vocabulary from a list of texts, separating words.',
      example: <<~EX
        texts = ["hello world", "good day"]
        vocabulary = GRYDRA.create_vocabulary(texts)
      EX
    },
    'GRYDRA.vectorize_text' => {
      description: 'Converts a text into a binary vector based on the presence of words in the vocabulary.',
      example: <<~EX
        vector = GRYDRA.vectorize_text("hello world", vocabulary)
      EX
    },
    'GRYDRA.normalize_with_vocabulary' => {
      description: 'Normalizes vectors generated with the vocabulary by dividing by the vocabulary size.',
      example: <<~EX
        vectors_norm = GRYDRA.normalize_with_vocabulary(vectors, vocabulary)
      EX
    },
    'GRYDRA.generate_example' => {
      description: 'Generates a functional code example with the library, with examples from 1 to 9.',
      example: <<~EX
        GRYDRA.generate_example(1)
      EX
    },
    'GRYDRA.suggest_structure' => {
      description: 'Automatically suggests a possible neural network structure based on the number of inputs and outputs.',
      example: <<~EX
        suggested_structure = GRYDRA.suggest_structure(3, 1)
      EX
    },
    'GRYDRA.convert_hashes_to_vectors' => {
      description: 'Converts an array of hashes (like JSON) to numerical arrays for training.',
      example: <<~EX
        data = [
          { name: "A", age: 20, vip: true },
          { name: "B", age: 30, vip: false }
        ]
        data_vectors = GRYDRA.convert_hashes_to_vectors(data, [:age, :vip])
      EX
    },
    'GRYDRA.summary_model' => {
      description: 'Displays the subnets, their structures, and activation functions of a loaded model to the console.',
      example: <<~EX
        GRYDRA.summary_model(model)
      EX
    },
    'GRYDRA.validate_model' => {
      description: 'Checks if a "model" is actually a compatible model.',
      example: <<~EX
        GRYDRA.validate_model(model)
      EX
    },
    'GRYDRA.test_all_normalizations' => {
      description: 'Tests training with :max and :zscore and shows the final error with each one.',
      example: <<~EX
        inputs = [[1], [2], [3]]
        outputs = [[2], [4], [6]]
        structure = [[1, 3, 1]]
        GRYDRA.test_all_normalizations(inputs, outputs, structure)
      EX
    }
  }
  # Function to display description and example of a method given class and method (strings)
  def self.describe_method(class_name, method_name)
    key = "#{class_name}.#{method_name}"
    info = METHOD_DESCRIPTIONS[key]
    if info
      puts "\e[1;3;5;37mDescription of #{key}:"
      puts info[:description]
      puts "
Example of use:"
      puts "#{info[:example]}\e[0m"
    else
      puts "\e[31;1mNo description found for method '#{key}'"
      puts "\e[31mMake sure to use the exact class and method name (as strings)"
      puts "\e[36mYou can call the method to verify: list_methods_available\e[0m"
    end
  end
  # Function that lists all documented public methods in METHOD_DESCRIPTIONS
  def self.list_methods_available
    puts "\e[1;3;5;37mDocumented public methods:"
    grouped = METHOD_DESCRIPTIONS.keys.group_by { |k| k.split('.').first }
    grouped.each do |class_name, methods|
      puts "  #{class_name}:"
      methods.each { |m| puts "    - #{m.split('.').last}" }
    end
    print "\e[0m"
  end
  def self.generate_example(num_example, filename = 'example', extension = 'rb', path = Dir.pwd)
    case num_example
    when 1
      content = <<~RUBY
        require 'grydra'
        # Training data
        training_data = [
          { name: "Company 1", num_employees: 5, is_new: false, site: true, label: 0 },
          { name: "Company 2", num_employees: 4, is_new: true, site: false, label: 0 },
          { name: "Company 3", num_employees: 4, is_new: false, site: false, label: 1 },
          { name: "Company 4", num_employees: 20, is_new: false, site: false, label: 1 },
          { name: "Company 5", num_employees: 60, is_new: false, site: false, label: 1 },
          { name: "Company 6", num_employees: 90, is_new: false, site: false, label: 0 },
          { name: "Company 7", num_employees: 33, is_new: true, site: false, label: 0 },
          { name: "Company 8", num_employees: 33, is_new: false, site: true, label: 0 },
          { name: "Company 9", num_employees: 15, is_new: false, site: false, label: 1 },
          { name: "Company 10", num_employees: 40, is_new: false, site: true, label: 0 },
          { name: "Company 11", num_employees: 3, is_new: false, site: false, label: 0 },
          { name: "Company 12", num_employees: 66, is_new: false, site: true, label: 0 },
          { name: "Company 13", num_employees: 15, is_new: true, site: false, label: 0 },
          { name: "Company 13", num_employees: 10, is_new: false, site: false, label: 1 },
          { name: "Company 13", num_employees: 33, is_new: false, site: false, label: 1 },
          { name: "Company 13", num_employees: 8, is_new: false, site: false, label: 1 },
        ]
        # Create the model with regularization and dropout
        model = GRYDRA::EasyNetwork.new(true, true)
        # Configure Adam optimizer
        model.configure_adam_optimizer(0.001, 0.9, 0.999)
        # Train with L2 regularization and dropout
        model.train_hashes(
          training_data,
          [:num_employees, :is_new, :site],
          :label,
          [[3, 4, 1]],
          0.05,
          12000,
          :max,
          lambda_l2: 0.001,
          dropout: true,
          dropout_rate: 0.3
        )
        # Save the trained model
        GRYDRA.save_model(model, "company_model_advanced")
        puts "Training completed with regularization and Adam optimizer."
      RUBY
    when 10
      content = <<~RUBY
        require 'grydra'
        # Example of cross-validation with hyperparameter search
        puts "Example of Cross-Validation and Hyperparameter Search"
        # Generate synthetic data
        synthetic_data = GRYDRA.generate_synthetic_data(200, 3, 0.1, 42)
        data_x = synthetic_data[:data]
        data_y = synthetic_data[:labels]
        # Define hyperparameter grid
        param_grid = [
          { rate: 0.01, epochs: 1000, lambda_l2: 0.001 },
          { rate: 0.05, epochs: 800, lambda_l2: 0.01 },
          { rate: 0.1, epochs: 500, lambda_l2: 0.001 },
          { rate: 0.02, epochs: 1200, lambda_l2: 0.005 }
        ]
        # Hyperparameter search with cross-validation
        best_result = GRYDRA.hyperparameter_search(data_x, data_y, param_grid) do |params, x, y|
          # Cross-validation for each configuration
          result_cv = GRYDRA.cross_validation(x, y, 5) do |train_x, train_y, test_x, test_y|
            # Create and train model
            model = GRYDRA::EasyNetwork.new(false)
            model.configure_adam_optimizer(params[:rate])
        #{'    '}
            model.train_numerical(
              train_x, train_y, [[4, 3, 1]],#{' '}
              params[:rate], params[:epochs], :max,
              lambda_l2: params[:lambda_l2],
              patience: 50
            )
        #{'    '}
            # Evaluate on test set
            predictions = model.predict_numerical(test_x)
            GRYDRA.mse(predictions.flatten, test_y.flatten)
          end
        #{'  '}
          result_cv[:average]
        end
        puts "\
 Best configuration found:"
        puts "Parameters: \#{best_result[:parameters]}"
        puts "Average CV Error: \#{best_result[:score].round(6)}"
        # Train final model with best parameters
        puts "\
 Training final model..."
        final_model = GRYDRA::EasyNetwork.new(true)
        final_model.configure_adam_optimizer(best_result[:parameters][:rate])
        # Split data for final training
        split = GRYDRA.split_data(data_x, data_y, 0.8, 42)
        final_model.train_numerical(
          split[:train_x], split[:train_y], [[4, 3, 1]],
          best_result[:parameters][:rate],#{' '}
          best_result[:parameters][:epochs],#{' '}
          :max,
          lambda_l2: best_result[:parameters][:lambda_l2]
        )
        # Evaluate final model
        evaluation = final_model.evaluate_model(split[:test_x], split[:test_y], [:mse, :mae])
        puts "\
 Evaluation of final model:"
        puts "MSE: \#{evaluation[:mse].round(6)}"
        puts "MAE: \#{evaluation[:mae].round(6)}"
        # Analyze gradients
        analysis = GRYDRA.analyze_gradients(final_model.network)
        puts "\
 Gradient analysis:"
        puts "Average: \#{analysis[:average].round(6)}"
        puts "Maximum: \#{analysis[:maximum].round(6)}"
        puts "Minimum: \#{analysis[:minimum].round(6)}"
        # Show architecture
        GRYDRA.plot_architecture_ascii(final_model.network)
      RUBY
    when 11
      content = <<~RUBY
        require 'grydra'
        # Example of advanced text processing with TF-IDF
        puts " Example of Advanced Text Processing"
        # Example corpus
        texts = [
          "the cat climbed the tree",
          "the dog ran through the park",
          "birds fly high",
          "the cat and the dog are friends",
          "the trees in the park are tall",
          "the dog barks at the cat",
          "birds sing in the trees"
        ]
        # Labels (0: animals, 1: nature)
        labels = [[0], [0], [1], [0], [1], [0], [1]]
        # Create advanced vocabulary with filtering
        vocabulary = GRYDRA.create_advanced_vocabulary(texts, 2, 50)
        puts "Vocabulary created: \#{vocabulary.size} words"
        puts "Words: \#{vocabulary.join(', ')}"
        # Calculate corpus frequencies for TF-IDF
        corpus_freqs = Hash.new(0)
        texts.each do |text|
          text.split.uniq.each { |word| corpus_freqs[word] += 1 }
        end
        # Vectorize texts using TF-IDF
        vectors_tfidf = texts.map do |text|
          GRYDRA.vectorize_text_tfidf(text, vocabulary, corpus_freqs)
        end
        puts "\
TF-IDF vectorization completed"
        puts "Vector dimension: \#{vectors_tfidf.first.size}"
        # Train model with TF-IDF vectors
        model = GRYDRA::EasyNetwork.new(true)
        model.configure_adam_optimizer(0.01)
        # Normalize TF-IDF vectors
        max_values = GRYDRA.calculate_max_values(vectors_tfidf, :max)
        vectors_norm = GRYDRA.normalize_multiple(vectors_tfidf, max_values, :max)
        # Train
        model.train_numerical(
          vectors_norm, labels, [[8, 4, 1]],#{' '}
          0.01, 2000, :max,
          lambda_l1: 0.001,
          dropout: true,
          dropout_rate: 0.2
        )
        # Test with new texts
        new_texts = [
          "the cat sleeps",
          "the trees are green",
          "the dog plays"
        ]
        puts "\
 Predictions for new texts:"
        new_texts.each do |text|
          vector_tfidf = GRYDRA.vectorize_text_tfidf(text, vocabulary, corpus_freqs)
          vector_norm = GRYDRA.normalize_multiple([vector_tfidf], max_values, :max)
          prediction = model.predict_numerical(vector_norm)
        #{'  '}
          category = prediction[0][0] > 0.5 ? "Nature" : "Animals"
          puts "'\#{text}' → \#{prediction[0][0].round(3)} (\#{category})"
        end
        # Show model analysis
        puts "\
 Model analysis:"
        evaluation = model.evaluate_model(vectors_norm, labels, [:mse, :mae])
        puts "MSE: \#{evaluation[:mse].round(6)}"
        puts "MAE: \#{evaluation[:mae].round(6)}"
        GRYDRA.plot_architecture_ascii(model.network)
      RUBY
    when 12
      content = <<~RUBY
        require 'grydra'
        # Example of advanced classification metrics
        puts " Example of Advanced Classification Metrics"
        # Generate binary classification data
        data_x = [
          [0.1, 0.2], [0.8, 0.9], [0.2, 0.1], [0.9, 0.8],
          [0.3, 0.4], [0.7, 0.6], [0.4, 0.3], [0.6, 0.7],
          [0.15, 0.25], [0.85, 0.75], [0.25, 0.15], [0.75, 0.85]
        ]
        data_y = [[0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1]]
        # Split data
        split = GRYDRA.split_data(data_x, data_y, 0.7, 42)
        # Train model
        model = GRYDRA::EasyNetwork.new(true)
        model.configure_adam_optimizer(0.1)
        model.train_numerical(
          split[:train_x], split[:train_y], [[3, 1]],#{' '}
          0.1, 1000, :max,
          lambda_l2: 0.01
        )
        # Make predictions
        predictions = model.predict_numerical(split[:test_x])
        pred_flat = predictions.flatten
        actual_flat = split[:test_y].flatten
        puts "\
 Predictions vs Actuals:"
        predictions.zip(split[:test_y]).each_with_index do |(pred, actual), i|
          puts "Sample \#{i+1}: Pred=\#{pred[0].round(3)}, Actual=\#{actual[0]}"
        end
        # Calculate advanced metrics
        puts "\
 Evaluation Metrics:"
        # Accuracy
        accuracy = GRYDRA.accuracy(pred_flat, actual_flat, 0.5)
        puts "Accuracy: \#{(accuracy * 100).round(2)}%"
        # Confusion matrix
        matrix = GRYDRA.confusion_matrix(pred_flat, actual_flat, 0.5)
        puts "\
Confusion Matrix:"
        puts "  TP: \#{matrix[:tp]}, FP: \#{matrix[:fp]}"
        puts "  TN: \#{matrix[:tn]}, FN: \#{matrix[:fn]}"
        # Precision, Recall, F1
        if matrix[:tp] + matrix[:fp] > 0
          precision = GRYDRA.precision(matrix[:tp], matrix[:fp])
          puts "Precision: \#{(precision * 100).round(2)}%"
        end
        if matrix[:tp] + matrix[:fn] > 0
          recall = GRYDRA.recall(matrix[:tp], matrix[:fn])
          puts "Recall: \#{(recall * 100).round(2)}%"
        #{'  '}
          if defined?(precision) && precision > 0 && recall > 0
            f1 = GRYDRA.f1(precision, recall)
            puts "F1-Score: \#{(f1 * 100).round(2)}%"
          end
        end
        # AUC-ROC
        auc = GRYDRA.auc_roc(pred_flat, actual_flat)
        puts "AUC-ROC: \#{auc.round(4)}"
        # MSE and MAE
        mse = GRYDRA.mse(pred_flat, actual_flat)
        mae = GRYDRA.mae(pred_flat, actual_flat)
        puts "\
Regression Metrics:"
        puts "MSE: \#{mse.round(6)}"
        puts "MAE: \#{mae.round(6)}"
        # Gradient analysis
        puts "\
 Gradient Analysis:"
        analysis = GRYDRA.analyze_gradients(model.network)
        puts "Average: \#{analysis[:average].round(6)}"
        puts "Deviation: \#{analysis[:deviation].round(6)}"
        puts "Range: [\#{analysis[:minimum].round(6)}, \#{analysis[:maximum].round(6)}]"
        # Show architecture
        GRYDRA.plot_architecture_ascii(model.network)
      RUBY
    else
      content = case num_example
      when 2
        <<~RUBY
          require 'grydra'
          model = nil
          model = GRYDRA.load_model("company_model_advanced")
          # New company data to evaluate
          new_data = [
            { name: "New Company A", num_employees: 12, is_new: true, site: true },
            { name: "New Company B", num_employees: 50, is_new: false, site: false },
            { name: "New Company C", num_employees: 7, is_new: false, site: false },
            { name: "New Company D", num_employees: 22, is_new: true, site: true }
          ]
          # Make predictions
          predictions = model.predict_hashes(new_data, [:num_employees, :is_new, :site])
          # Show results
          new_data.each_with_index do |company, i|
            prediction = predictions[i].first.round(3)
            puts "Company: \#{company[:name]} → Prediction: \#{prediction} (\#{prediction >= 0.5 ? 'Label 1 (Yes)' : 'Label 0 (No)'})"
          end
        RUBY
      when 3
        <<~RUBY
          require 'grydra'
          # Create main network
          network = GRYDRA::MainNetwork.new #No parameters, just don't print epochs or plot
          # Add a subnet with structure [2 inputs, 2 hidden, 1 output]
          network.add_subnet([2, 3, 1], [:tanh, :tanh])
          network.add_subnet([2, 4, 1], [:sigmoid, :sigmoid])
          # XOR data
          inputs = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
          ]
          outputs = [
            [0],
            [1],
            [1],
            [0]
          ]
          # Training
          epochs = 6000
          learning_rate = 0.9
          network.train_subnets(
            [
              {input: inputs, output: outputs},
              {input: inputs, output: outputs},
            ],
            learning_rate,#{' '}
            epochs,
            batch_size: 1, #Number of data points to train simultaneously
            patience: 100, #If results don't improve for (n) epochs, then stop#{' '}
            decay: 0.995 #Number by which it multiplies#{' '}
          )
          # Evaluation
          puts "\
XOR Evaluation:"
          inputs.each do |input|
            output = network.combine_results(input)
            puts "Input: \#{input} => Output: \#{output.map { |v| v.round(3)  }}" #>0.5 in this case would be 1#{' '}
          end
        RUBY
      when 4
        <<~RUBY
          require 'grydra'
          # Original data (Celsius and Fahrenheit temperatures)
          data_in = [[0], [10], [20], [30], [40], [50], [-10], [-20], [100], [-30], [-5], [-40]]
          data_out = [[32], [50], [68], [86], [104], [122], [14], [-4], [212], [-22], [23], [-40]]
          # Find maximum values to normalize (:max method)
          max_in = GRYDRA.calculate_max_values(data_in, :max)  # {0 => value}
          max_out = GRYDRA.calculate_max_values(data_out, :max)  # {0 => value}
          # Normalize data
          data_in_no = GRYDRA.normalize_multiple(data_in, max_in, :max)
          data_out_no = GRYDRA.normalize_multiple(data_out, max_out, :max)
          # Create main network
          main_network = GRYDRA::MainNetwork.new(false, true) #Doesn't print errors, but plots
          # Add subnets
          main_network.add_subnet([1, 4, 1], [:sigmoid, :tanh])
          main_network.add_subnet([1, 3, 1], [:relu, :tanh])
          main_network.add_subnet([1, 2, 1], [:tanh, :tanh])
          puts "Training subnets..."
          main_network.train_subnets(
            [
              { input: data_in_no, output: data_out_no },
              { input: data_in_no, output: data_out_no },
              { input: data_in_no, output: data_out_no }
            ],
            0.2,    # learning rate
            25000,   # epochs
            batch_size: 5,
            patience: 500,
            decay: 0.995
          )
          puts "\
Enter Celsius degrees separated by space:"
          print "<< "
          user_input = gets.chomp.split.map(&:to_f)
          # Normalize inputs
          user_input_no = user_input.map { |e| [e] }
          user_input_no = GRYDRA.normalize_multiple(user_input_no, max_in, :max)
          puts "\
Combined results:"
          user_input_no.each_with_index do |input_norm, i|
            prediction_norm = main_network.combine_results(input_norm)
            prediction = [prediction_norm[0] * max_out[0]]#{'  '}
            puts "\#{user_input[i]} °C : \#{prediction[0].round(2)} °F"
          end
        RUBY
      when 5
        <<~RUBY
          require 'grydra'
          =begin
          IMPORTANT NOTE:
          If using zscore normalization, we must be more meticulous and it's recommended#{' '}
          to constantly lower and raise the learning rate, because with zscore once it reaches#{' '}
          the expected results it will stay around that margin. The rate can be lowered and raised#{' '}
          as much as desired, for example:
          0.003, 0.3, 0.9, 0.2, 0.223, 0.00008
          =end
          # Create instance of EasyNetwork class
          network = GRYDRA::EasyNetwork.new(true, true)  # true to print epochs and another true to plot
          # Original data (Celsius and Fahrenheit temperatures)
          data_in = [[0], [10], [20], [30], [40], [50], [-10], [-20], [100], [-30], [-5], [-40]]
          data_out = [[32], [50], [68], [86], [104], [122], [14], [-4], [212], [-22], [23], [-40]]
          # Define subnet structures (hidden layers)
          structures = [
            [2, 4, 1],  # hidden layer with 4 neurons, 1 output
            [1, 3, 1],  # another subnet, 3 hidden neurons
            [2, 7, 1]   # and another smaller one
          ]
          puts "Training network..."
          network.train_numerical(data_in, data_out, structures, 0.5, 30000, :zscore)#{' '}
          #network.train_numerical(data_in, data_out, structures, 0.5, 30000, :max)
          puts "\
Enter Celsius degrees separated by space:"
          print "<< "
          user_input = gets.chomp.split.map(&:to_f).map { |v| [v] }
          predictions = network.predict_numerical(user_input, :zscore)
          #predictions = network.predict_numerical(user_input, :max)#{' '}
          puts "\
Results:"
          user_input.each_with_index do |input, i|
            f = predictions[i][0]
            puts "\#{input[0]} °C : \#{f.round(2)} °F"
          end
        RUBY
      when 6
        <<~RUBY
          require 'grydra'
          # Create input and output data
          data_input = [
            [170, 25],
            [160, 30],
            [180, 22],
            [150, 28],
            [175, 24]
          ]
          # Weight corresponding to each person (kg)
          data_output = [
            [65],
            [60],
            [75],
            [55],
            [70]
          ]
          # Define subnet structures
          # Each subnet has: 2 inputs → 3 hidden neurons → 1 output
          structures = [
            [3, 1],
            [4, 1]
          ]
          # Create network using the easy interface
          network = GRYDRA::EasyNetwork.new(true)  # true to print only error per epoch
          # Train the network
          network.train_numerical(
            data_input,
            data_output,
            structures,
            0.05,        # learning rate
            15000,       # epochs
            :max         # normalization type
          )
          # Predict for a new individual
          new_data = [[172, 26]]
          predictions = network.predict_numerical(new_data, :max)
          puts "\
Result:"
          puts "Height: \#{new_data[0][0]}, Age: \#{new_data[0][1]} ⇒ Estimated weight: \#{predictions[0][0].round(2)} kg"
        RUBY
      when 7
        <<~RUBY
          require 'grydra'
          # Training data: [height (cm), age (years)]
          data_input = [
            [170, 25],
            [160, 30],
            [180, 22],
            [175, 28],
            [165, 35],
            [155, 40],
            [185, 20]
          ]
          # Real weight in kg
          data_output = [
            [65],
            [60],
            [75],
            [70],
            [62],
            [58],
            [80]
          ]
          # Structures for subnets: hidden layers and output
          structures = [
            [4, 1],#{'   '}
            [3, 1],
            [6, 1],
            [2, 1]
          ]
          # Create Easy Network
          network = GRYDRA::EasyNetwork.new(true, true) # true to see training progress and another true to plot
          # We will assign 'sigmoid' activation to the last layer to limit output between 0 and 1
          # Adjust internally in the add_subnet method
          structures.each do |structure|
            # Define activations: hidden with :tanh, output with :sigmoid
            activations = Array.new(structure.size - 1, :tanh) + [:sigmoid]
            network.network.add_subnet([data_input.first.size, *structure], activations)
          end
          # :max normalization (easy to denormalize)
          network.train_numerical(data_input, data_output, structures, 0.01, 10000, :max)
          # New sample to predict: height=172 cm, age=26 years
          new_data = [[172, 26]]
          # Make prediction (normalized internally)
          predictions = network.predict_numerical(new_data, :max)
          # Predictions are already denormalized by EasyNetwork, just round them
          puts "Predicted weight (kg) for height \#{new_data[0][0]} cm and age \#{new_data[0][1]} years:"
          print predictions.map { |p| p[0].round(2) }
          GRYDRA.save_model(network, "average_weight")
        RUBY
      when 8
        <<~RUBY
          # predict.rb
          require 'grydra'
          # Load the previously saved model
          model = GRYDRA.load_model("average_weight")
          # Input data to predict: height=172 cm, age=26 years
          new_data = [[172, 26]]
          # Normalization used (can be :max or :zscore, depending on training)
          normalization = :max
          # Make prediction
          predictions = model.predict_numerical(new_data, normalization)
          puts "Predicted weight (kg) for height \#{new_data[0][0]} cm and age \#{new_data[0][1]} years:"
          puts predictions.map { |p| p[0].round(2) }
        RUBY
      when 9
        <<~RUBY
          #Program to determine product price
          require 'grydra'
          #product cost will be in dollars
          data_input = [
            [10, 0], #<-- Number of products per company and (0 and 1) if is vip or not#{' '}
            [15, 0],
            [8,  1],
            [20, 1],
            [12, 0],
            [30, 1],
            [25, 1],
            [5,  0],
            [18, 0],
            [40, 1]
          ]
          #Price in dollars
          data_output = [
            [20],
            [28],
            [25],
            [45],
            [22],
            [60],
            [50],
            [12],
            [32],
            [80]
          ]
          #Data normalization (common data --> Vector data)
          max_in = GRYDRA.calculate_max_values(data_input, :max) #We will use max normalization, although zscore is possible
          max_out = GRYDRA.calculate_max_values(data_output, :max)
          data_in_no = GRYDRA.normalize_multiple(data_input, max_in, :max) #By default uses max, so :max is optional
          data_out_no = GRYDRA.normalize_multiple(data_output, max_out)
          #Create the network
          network = GRYDRA::MainNetwork.new
          #We need to add subnets to our network
          network.add_subnet([2, 4, 1], [:relu, :tanh])#{' '}
          network.add_subnet([2, 3, 1], [:tanh, :tanh])#{' '}
          puts "Training subnets"
          network.train_subnets(
            [
              {input: data_in_no, output: data_out_no},
              {input: data_in_no, output: data_out_no}
            ],
            0.2, #Learning rate
            20000, #Number of epochs
            batch_size: 3, #Means it will analyze 3 data points at a time
            patience: 500, #If results don't improve in these epochs then this network stops
            decay: 0.995 #number by which it multiplies
          )
          puts "Enter new values to predict the product price eg: (12 0)"
          values = gets.chomp.strip.split.map(&:to_f)
          input_norm = GRYDRA.normalize_multiple([values], max_in, :max)[0]
          prediction = network.combine_results(input_norm)
          prediction_denorm = prediction[0] * max_out[0]
          puts "Approximate price in dollars is $#{prediction_denorm.round(2)}"
        RUBY
      else
        puts "\e[1;35mPossible examples are from 1 to 12\e[0m"
        return
      end
    end
    return unless num_example.between?(1, 12)
    File.write(File.join(path, "#{filename}.#{extension}"), content)
    puts "Example generated and saved to \e[33m#{File.join(path, filename)}\e[0m"
  end
  def self.suggest_structure(inputs, outputs = 1)
    hidden = [(inputs + outputs) * 2, (inputs + outputs)].uniq
    [[inputs, *hidden, outputs]]
  end
  def self.plot_error(errors, print_every = 5, bar_width = 40, delta_min = 0.001)
    max_error = errors.max
    first_error = errors.first
    puts "
Error graph by epoch"
    puts '-' * (bar_width + 40)
    last_printed = nil
    errors.each_with_index do |error, i|
      epoch = i + 1
      next unless epoch == 1 || epoch == errors.size || epoch % print_every == 0
      if last_printed && (last_printed - error).abs < delta_min && epoch != errors.size
        # If the difference from the last printed is less than delta, skip
        next
      end
      bar_length = [(bar_width * error / max_error).round, 1].max
      bar = '=' * bar_length
      improvement_pct = ((first_error - error) / first_error.to_f) * 100
      improvement_str = improvement_pct >= 0 ? "+#{improvement_pct.round(2)}%" : "#{improvement_pct.round(2)}%"
      puts "Epoch #{epoch.to_s.ljust(4)} | #{bar.ljust(bar_width)} | Error: #{error.round(6)} | Improvement: #{improvement_str}"
      last_printed = error
    end
    puts '-' * (bar_width + 40)
    puts "Initial error: #{first_error.round(6)}, Final error: #{errors.last.round(6)}
"
  end
  def self.convert_hashes_to_vectors(array_hashes, keys)
    array_hashes.map do |hash|
      keys.map do |k|
        if hash[k]
          if hash[k] == true
            1.0
          else
            hash[k] == false ? 0.0 : hash[k].to_f
          end
        else
          0.0
        end
      end
    end
  end
  def self.summary_model(model, input_test = nil)
    puts "
\e[1;36mModel summary:\e[0m"
    # In case it's a wrapper like EasyNetwork
    model = model.network if model.respond_to?(:network) && model.network.respond_to?(:subnets)
    if model.respond_to?(:subnets)
      model.subnets.each_with_index do |subnet, i|
        puts "
 Subnet ##{i + 1}:"
        structure = subnet.layers.map { |l| l.neurons.size }
        hidden_activations = subnet.layers[0...-1].map(&:activation)
        output_function = subnet.layers.last.activation
        puts "  - Structure: #{structure.inspect}"
        puts "  - Hidden activations: #{hidden_activations.inspect}"
        puts "  - Output function: #{output_function.inspect}"
        next unless input_test
        begin
          output = subnet.calculate_outputs(input_test)
          puts "  - Numerical output with input #{input_test.inspect}: #{output.inspect}"
        rescue StandardError => e
          puts "  - Error calculating output: #{e.message}"
        end
      end
    else
      GRYDRA.validate_model(model)
    end
  end
  def self.test_all_normalizations(inputs, outputs, structures)
    %i[max zscore].each do |type|
      puts "
Testing normalization: #{type}"
      network = GRYDRA::EasyNetwork.new(false)
      final_error = network.train_numerical(inputs, outputs, structures, 0.1, 5000, type)
      puts " Final error: #{final_error}"
    end
  end
  def self.validate_model(model)
    if model.nil?
      puts "\e[31mError: model is nil\e[0m"
    elsif model.is_a?(GRYDRA::EasyNetwork) || model.is_a?(GRYDRA::MainNetwork)
      puts "\e[32mValid model of type #{model.class}\e[0m"
    else
      puts "\e[33mWarning: The loaded model is not a known instance (#{model.class})\e[0m"
    end
  end
end
