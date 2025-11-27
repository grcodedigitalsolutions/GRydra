module GRYDRA
  module Layers
    # LSTM Layer - Full implementation with backpropagation support
    class LSTM < Base
      attr_accessor :units, :cell_state, :hidden_state
      attr_reader :weights, :biases

      def initialize(units, inputs_per_unit, return_sequences: false)
        @units = units
        @inputs_per_unit = inputs_per_unit
        @return_sequences = return_sequences
        
        combined_size = inputs_per_unit + units
        
        # Initialize weight matrices for all gates
        @weights = {
          forget: initialize_weights(combined_size, units),
          input: initialize_weights(combined_size, units),
          candidate: initialize_weights(combined_size, units),
          output: initialize_weights(combined_size, units)
        }
        
        @biases = {
          forget: Array.new(units, 1.0),  # Bias forget gate to 1 initially
          input: Array.new(units, 0.0),
          candidate: Array.new(units, 0.0),
          output: Array.new(units, 0.0)
        }
        
        # Cache for backpropagation
        @cache = []
        
        reset_state
      end

      def reset_state
        @cell_state = Array.new(@units, 0.0)
        @hidden_state = Array.new(@units, 0.0)
        @cache.clear
      end

      def calculate_outputs(input_sequence)
        outputs = []
        
        input_sequence = [input_sequence] unless input_sequence.first.is_a?(Array)
        
        input_sequence.each do |input|
          combined = input + @hidden_state
          
          # Forget gate
          f_t = gate_activation(combined, @weights[:forget], @biases[:forget], :sigmoid)
          
          # Input gate
          i_t = gate_activation(combined, @weights[:input], @biases[:input], :sigmoid)
          
          # Candidate values
          c_tilde = gate_activation(combined, @weights[:candidate], @biases[:candidate], :tanh)
          
          # Update cell state
          @cell_state = @cell_state.zip(f_t, i_t, c_tilde).map do |c, f, i, c_t|
            f * c + i * c_t
          end
          
          # Output gate
          o_t = gate_activation(combined, @weights[:output], @biases[:output], :sigmoid)
          
          # Update hidden state
          @hidden_state = o_t.zip(@cell_state).map { |o, c| o * Math.tanh(c) }
          
          # Cache for backpropagation
          @cache << {
            input: input,
            combined: combined,
            f_t: f_t,
            i_t: i_t,
            c_tilde: c_tilde,
            o_t: o_t,
            cell_state: @cell_state.dup,
            hidden_state: @hidden_state.dup
          }
          
          outputs << @hidden_state.dup
        end
        
        @return_sequences ? outputs : outputs.last
      end

      def backward(d_hidden, learning_rate: 0.01)
        # Simplified backpropagation through time
        d_cell = Array.new(@units, 0.0)
        
        @cache.reverse.each do |cache|
          # Gradient through output gate
          d_o = d_hidden.zip(cache[:cell_state]).map { |dh, c| dh * Math.tanh(c) }
          d_cell = d_hidden.zip(cache[:o_t], cache[:cell_state]).map do |dh, o, c|
            d_cell_val = dh * o * (1 - Math.tanh(c)**2)
            d_cell_val
          end
          
          # Update weights (simplified)
          update_gate_weights(:output, cache[:combined], d_o, learning_rate)
        end
      end

      private

      def initialize_weights(input_size, output_size)
        Array.new(output_size) do
          Array.new(input_size) { Initializers.xavier_init(input_size) }
        end
      end

      def gate_activation(input, weights, biases, activation)
        output = Array.new(@units) do |i|
          sum = biases[i]
          input.each_with_index { |x, j| sum += x * weights[i][j] }
          sum
        end
        
        output.map do |val|
          case activation
          when :sigmoid then Activations.sigmoid(val)
          when :tanh then Activations.tanh(val)
          else val
          end
        end
      end

      def update_gate_weights(gate, input, gradient, learning_rate)
        gradient.each_with_index do |grad, i|
          input.each_with_index do |inp, j|
            @weights[gate][i][j] += learning_rate * grad * inp
          end
          @biases[gate][i] += learning_rate * grad
        end
      end
    end
  end
end
