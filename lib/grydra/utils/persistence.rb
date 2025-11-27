module GRYDRA
  module Utils
    module Persistence
      # Save model to file
      def self.save_model(model, name, path = Dir.pwd, vocabulary = nil)
        file_path = File.join(path, "#{name}.net")
        File.open(file_path, 'wb') { |f| Marshal.dump(model, f) }
        puts "Model saved to '#{file_path}'"

        save_vocabulary(vocabulary, name, path) if vocabulary
      end

      # Load model from file
      def self.load_model(name, path = Dir.pwd)
        model = nil
        file_path = File.join(path, "#{name}.net")
        File.open(file_path, 'rb') { |f| model = Marshal.load(f) }
        model
      end

      # Save vocabulary to file
      def self.save_vocabulary(vocabulary, name, path = Dir.pwd)
        file_path = File.join(path, "#{name}_vocab.bin")
        File.open(file_path, 'wb') { |f| Marshal.dump(vocabulary, f) }
        puts "Vocabulary saved to '#{file_path}'"
      end

      # Load vocabulary from file
      def self.load_vocabulary(name, path = Dir.pwd)
        vocabulary = nil
        file_path = File.join(path, "#{name}_vocab.bin")
        File.open(file_path, 'rb') { |f| vocabulary = Marshal.load(f) }
        vocabulary
      end

      # Validate model
      def self.validate_model(model)
        if model.nil?
          puts "ERROR: model is nil"
          false
        elsif model.is_a?(Networks::EasyNetwork) || model.is_a?(Networks::MainNetwork)
          puts "Valid model of type #{model.class}"
          true
        else
          puts "WARNING: The loaded model is not a known instance (#{model.class})"
          false
        end
      end

      # Summary of model
      def self.summary_model(model, input_test: nil)
        puts "\nModel Summary:"
        puts "=" * 60

        model = model.network if model.respond_to?(:network) && model.network.respond_to?(:subnets)

        if model.respond_to?(:subnets)
          total_params = 0
          
          model.subnets.each_with_index do |subnet, i|
            puts "\nSubnet ##{i + 1}:"
            structure = subnet.layers.map { |l| l.neurons.size }
            hidden_activations = subnet.layers[0...-1].map(&:activation)
            output_function = subnet.layers.last.activation

            subnet_params = subnet.layers.sum do |layer|
              layer.neurons.sum { |n| n.weights.size + 1 }
            end
            total_params += subnet_params

            puts "  Structure: #{structure.inspect}"
            puts "  Hidden activations: #{hidden_activations.inspect}"
            puts "  Output activation: #{output_function.inspect}"
            puts "  Parameters: #{subnet_params}"

            if input_test
              begin
                output = subnet.calculate_outputs(input_test)
                puts "  Test output: #{output.map { |v| v.round(4) }.inspect}"
              rescue StandardError => e
                puts "  Error calculating output: #{e.message}"
              end
            end
          end
          
          puts "\n" + "=" * 60
          puts "Total parameters: #{total_params}"
        else
          validate_model(model)
        end
      end
    end
  end
end
