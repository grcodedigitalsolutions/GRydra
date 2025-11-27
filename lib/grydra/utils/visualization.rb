module GRYDRA
  module Utils
    module Visualization
      # Plot error graph
      def self.plot_error(errors, print_every: 5, bar_width: 40, delta_min: 0.001)
        max_error = errors.max
        first_error = errors.first

        puts "\nTraining Error History"
        puts '=' * (bar_width + 40)

        last_printed = nil
        errors.each_with_index do |error, i|
          epoch = i + 1
          next unless epoch == 1 || epoch == errors.size || epoch % print_every == 0

          if last_printed && (last_printed - error).abs < delta_min && epoch != errors.size
            next
          end

          bar_length = [(bar_width * error / max_error).round, 1].max
          bar = '=' * bar_length
          improvement_pct = ((first_error - error) / first_error.to_f) * 100
          improvement_str = improvement_pct >= 0 ? "+#{improvement_pct.round(2)}%" : "#{improvement_pct.round(2)}%"

          puts "Epoch #{epoch.to_s.ljust(4)} | #{bar.ljust(bar_width)} | Error: #{error.round(6)} | Improvement: #{improvement_str}"
          last_printed = error
        end

        puts '=' * (bar_width + 40)
        puts "Initial error: #{first_error.round(6)}, Final error: #{errors.last.round(6)}"
        improvement = ((first_error - errors.last) / first_error * 100).round(2)
        puts "Total improvement: #{improvement}%\n"
      end

      # Plot architecture in ASCII
      def self.plot_architecture_ascii(model)
        puts "\nNetwork Architecture:"
        puts '=' * 60

        if model.respond_to?(:subnets)
          model.subnets.each_with_index do |subnet, idx|
            puts "\n  Subnet #{idx + 1}:"
            plot_individual_network(subnet)
          end
        else
          plot_individual_network(model)
        end

        puts '=' * 60
      end

      def self.plot_individual_network(network)
        network.layers.each_with_index do |layer, i|
          neurons = layer.neurons.size
          activation = layer.activation || :linear
          params = layer.neurons.sum { |n| n.weights.size + 1 }

          symbols = if neurons <= 10
                      'O' * neurons
                    else
                      'O' * 8 + "... (#{neurons} neurons)"
                    end

          puts "  Layer #{i + 1}: #{symbols}"
          puts "           Neurons: #{neurons}, Activation: #{activation}, Parameters: #{params}"
          puts "           |" unless i == network.layers.size - 1
          puts "           v" unless i == network.layers.size - 1
        end
      end

      # Analyze gradients
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
    end
  end
end
