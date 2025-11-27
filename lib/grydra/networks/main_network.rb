module GRYDRA
  module Networks
    # Main Network with multiple subnets
    class MainNetwork
      attr_accessor :subnets

      def initialize(print_epochs: false, plot: false)
        @subnets = []
        @print_epochs = print_epochs
        @plot = plot
      end

      def add_subnet(structure, activations = nil)
        @subnets << NeuralNetwork.new(structure, print_epochs: @print_epochs, plot: @plot, activations: activations)
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
  end
end
