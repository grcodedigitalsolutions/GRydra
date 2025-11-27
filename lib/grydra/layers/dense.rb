module GRYDRA
  module Layers
    # Dense (Fully Connected) Layer
    class Dense < Base
      attr_accessor :neurons, :activation

      def initialize(num_neurons, inputs_per_neuron, activation = :tanh)
        @activation = activation
        @neurons = Array.new(num_neurons) { Networks::Neuron.new(inputs_per_neuron, activation) }
      end

      def calculate_outputs(inputs, apply_dropout = false, dropout_rate = 0.5)
        @neurons.map { |neuron| neuron.calculate_output(inputs, apply_dropout, dropout_rate) }
      end
    end
  end
end
