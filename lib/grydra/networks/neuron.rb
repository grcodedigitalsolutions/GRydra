module GRYDRA
  module Networks
    # Individual Neuron class
    class Neuron
      attr_accessor :weights, :bias, :output, :delta

      def initialize(inputs, activation = :tanh)
        unless inputs.is_a?(Integer) && inputs > 0
          raise ArgumentError, 'Number of inputs must be a positive integer'
        end

        @weights = case activation
                   when :relu, :leaky_relu
                     Array.new(inputs) { Initializers.he_init(inputs) }
                   else
                     Array.new(inputs) { Initializers.xavier_init(inputs) }
                   end

        @bias = case activation
                when :relu, :leaky_relu
                  Initializers.he_init(inputs)
                else
                  Initializers.xavier_init(inputs)
                end

        @output = 0
        @delta = 0
        @activation = activation
        @sum = 0
        @dropout_mask = nil
      end

      def calculate_output(inputs, apply_dropout = false, dropout_rate = 0.5)
        unless inputs.is_a?(Array) && inputs.all? { |e| e.is_a?(Numeric) }
          raise ArgumentError, 'Inputs must be an array of numbers'
        end

        if @weights.size != inputs.size
          raise ArgumentError, "Error: inputs (#{inputs.size}) do not match weights (#{@weights.size})"
        end

        @sum = @weights.zip(inputs).map { |weight, input| weight * input }.sum + @bias
        @output = apply_activation(@sum)

        if apply_dropout
          @dropout_mask = rand < dropout_rate ? 0 : 1 / (1 - dropout_rate)
          @output *= @dropout_mask
        end

        @output
      end

      def derivative_activation
        case @activation
        when :tanh then Activations.derivative_tanh(@output)
        when :relu then Activations.derivative_relu(@output)
        when :sigmoid then Activations.derivative_sigmoid(@sum)
        when :leaky_relu then Activations.derivative_leaky_relu(@sum)
        when :swish then Activations.derivative_swish(@sum)
        when :gelu then Activations.derivative_gelu(@sum)
        else 1
        end
      end

      private

      def apply_activation(value)
        case @activation
        when :tanh then Activations.tanh(value)
        when :relu then Activations.relu(value)
        when :sigmoid then Activations.sigmoid(value)
        when :leaky_relu then Activations.leaky_relu(value)
        when :swish then Activations.swish(value)
        when :gelu then Activations.gelu(value)
        else value
        end
      end
    end
  end
end
