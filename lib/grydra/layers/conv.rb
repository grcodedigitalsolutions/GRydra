module GRYDRA
  module Layers
    # Convolutional Layer - Full 2D implementation
    class Conv < Base
      attr_accessor :filters, :kernel_size, :stride, :padding, :weights, :biases

      def initialize(filters, kernel_size, stride: 1, padding: 0, activation: :relu, input_channels: 1)
        @filters = filters
        @kernel_size = kernel_size
        @stride = stride
        @padding = padding
        @activation = activation
        @input_channels = input_channels
        
        # Initialize weights using He initialization for ReLU
        @weights = Array.new(filters) do
          Array.new(input_channels) do
            Array.new(kernel_size) do
              Array.new(kernel_size) { Initializers.he_init(kernel_size * kernel_size) }
            end
          end
        end
        @biases = Array.new(filters, 0.0)
        @output_cache = nil
      end

      def calculate_outputs(input)
        # Input shape: [channels, height, width]
        channels, height, width = input_shape(input)
        
        # Apply padding if needed
        padded_input = apply_padding(input) if @padding > 0
        padded_input ||= input
        
        # Calculate output dimensions
        out_height = ((height + 2 * @padding - @kernel_size) / @stride) + 1
        out_width = ((width + 2 * @padding - @kernel_size) / @stride) + 1
        
        # Perform convolution
        output = Array.new(@filters) do |f|
          Array.new(out_height) do |i|
            Array.new(out_width) do |j|
              convolve_at_position(padded_input, f, i, j)
            end
          end
        end
        
        @output_cache = output
        output
      end

      private

      def input_shape(input)
        if input.is_a?(Array) && input.first.is_a?(Array) && input.first.first.is_a?(Array)
          [input.size, input.first.size, input.first.first.size]
        else
          # Flatten input assumed
          [1, 1, input.size]
        end
      end

      def apply_padding(input)
        channels, height, width = input_shape(input)
        padded = Array.new(channels) do |c|
          Array.new(height + 2 * @padding) do |i|
            Array.new(width + 2 * @padding) do |j|
              if i < @padding || i >= height + @padding || j < @padding || j >= width + @padding
                0.0
              else
                input[c][i - @padding][j - @padding]
              end
            end
          end
        end
        padded
      end

      def convolve_at_position(input, filter_idx, out_i, out_j)
        sum = @biases[filter_idx]
        
        @input_channels.times do |c|
          @kernel_size.times do |ki|
            @kernel_size.times do |kj|
              i = out_i * @stride + ki
              j = out_j * @stride + kj
              sum += input[c][i][j] * @weights[filter_idx][c][ki][kj]
            end
          end
        end
        
        apply_activation(sum)
      end

      def apply_activation(value)
        case @activation
        when :relu then Activations.relu(value)
        when :tanh then Activations.tanh(value)
        when :sigmoid then Activations.sigmoid(value)
        when :leaky_relu then Activations.leaky_relu(value)
        else value
        end
      end
    end
  end
end
