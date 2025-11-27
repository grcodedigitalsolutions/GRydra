module GRYDRA
  module Networks
    # Neural Network class
    class NeuralNetwork
      attr_accessor :layers, :history_error

      def initialize(structure, print_epochs: false, plot: false, activations: nil)
        @print_epochs = print_epochs
        @plot = plot
        @layers = []
        @history_error = []
        @optimizer = nil

        activations ||= Array.new(structure.size - 1, :tanh)

        structure.each_cons(2).with_index do |(inputs, outputs), i|
          @layers << Layers::Dense.new(outputs, inputs, activations[i])
        end
      end

      def use_adam_optimizer(alpha = 0.001, beta1 = 0.9, beta2 = 0.999)
        @optimizer = Optimizers::Adam.new(alpha: alpha, beta1: beta1, beta2: beta2)
      end

      def calculate_outputs(inputs, apply_dropout = false, dropout_rate = 0.5)
        unless inputs.is_a?(Array) && inputs.all? { |e| e.is_a?(Numeric) }
          raise ArgumentError, 'Inputs must be an array of numbers'
        end

        @layers.inject(inputs) { |outputs, layer| layer.calculate_outputs(outputs, apply_dropout, dropout_rate) }
      end

      # Training with mini-batch, early stopping, decay learning rate, regularization and validation
      def train(data_input, data_output, learning_rate, epochs, 
                error_threshold: nil, batch_size: 1, patience: 50, decay: 0.95, 
                lambda_l1: 0, lambda_l2: 0, dropout: false, dropout_rate: 0.5, 
                validation_split: 0.0)
        best_error = Float::INFINITY
        patience_counter = 0
        
        # Split validation data if requested
        if validation_split > 0
          split_idx = (data_input.size * (1 - validation_split)).to_i
          indices = (0...data_input.size).to_a.shuffle
          train_indices = indices[0...split_idx]
          val_indices = indices[split_idx..-1]
          
          train_input = train_indices.map { |i| data_input[i] }
          train_output = train_indices.map { |i| data_output[i] }
          val_input = val_indices.map { |i| data_input[i] }
          val_output = val_indices.map { |i| data_output[i] }
        else
          train_input = data_input
          train_output = data_output
          val_input = nil
          val_output = nil
        end

        epochs.times do |epoch|
          error_total = 0
          error_regularization = 0

          # Shuffle training data
          indices = (0...train_input.size).to_a.shuffle
          train_input = indices.map { |i| train_input[i] }
          train_output = indices.map { |i| train_output[i] }

          data_input.each_slice(batch_size).with_index do |batch_inputs, batch_idx|
            batch_outputs_real = data_output[batch_idx * batch_size, batch_size]

            batch_inputs.zip(batch_outputs_real).each do |input, output_real|
              outputs = calculate_outputs(input, dropout, dropout_rate)
              errors = outputs.zip(output_real).map { |output, real| real - output }
              error_total += errors.map { |e| e**2 }.sum / errors.size

              # Calculate regularization penalty
              if lambda_l1 > 0 || lambda_l2 > 0
                @layers.each do |layer|
                  layer.neurons.each do |neuron|
                    error_regularization += Regularization.l1_regularization(neuron.weights, lambda_l1) if lambda_l1 > 0
                    error_regularization += Regularization.l2_regularization(neuron.weights, lambda_l2) if lambda_l2 > 0
                  end
                end
              end

              # Backpropagation
              backpropagate(input, errors, learning_rate, lambda_l1, lambda_l2)
            end
          end

          error_total += error_regularization

          # Early stopping check
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

        Utils::Visualization.plot_error(@history_error) if @plot
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

      private

      def backpropagate(input, errors, learning_rate, lambda_l1, lambda_l2)
        # Output layer
        @layers.last.neurons.each_with_index do |neuron, i|
          neuron.delta = errors[i] * neuron.derivative_activation
        end

        # Hidden layers
        (@layers.size - 2).downto(0) do |i|
          @layers[i].neurons.each_with_index do |neuron, j|
            sum_deltas = @layers[i + 1].neurons.sum { |n| n.weights[j] * n.delta }
            neuron.delta = sum_deltas * neuron.derivative_activation
          end
        end

        # Update weights and biases
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

              # Apply regularization
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
  end
end
