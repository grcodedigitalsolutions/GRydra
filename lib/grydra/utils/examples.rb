module GRYDRA
  module Utils
    module Examples
      # Suggest network structure
      def self.suggest_structure(inputs, outputs = 1)
        hidden = [(inputs + outputs) * 2, (inputs + outputs)].uniq
        [[inputs, *hidden, outputs]]
      end

      # Test all normalization methods
      def self.test_all_normalizations(inputs, outputs, structures)
        %i[max zscore].each do |type|
          puts "\n🧪 Testing normalization: #{type}"
          network = Networks::EasyNetwork.new(false)
          network.train_numerical(inputs, outputs, structures, 0.1, 5000, type)
          puts "  ✓ Training completed with #{type} normalization"
        end
      end

      # Generate example code
      def self.generate_example(num_example, filename = 'example', extension = 'rb', path = Dir.pwd)
        content = get_example_content(num_example)
        
        unless content
          puts "\e[1;35m⚠️  Available examples are from 1 to 12\e[0m"
          return
        end

        File.write(File.join(path, "#{filename}.#{extension}"), content)
        puts "✅ Example generated and saved to \e[33m#{File.join(path, filename)}.#{extension}\e[0m"
      end

      private

      def self.get_example_content(num)
        examples = {
          1 => example_1_basic_training,
          2 => example_2_load_and_predict,
          3 => example_3_xor_problem,
          4 => example_4_temperature_conversion,
          5 => example_5_zscore_normalization,
          6 => example_6_weight_prediction,
          7 => example_7_advanced_weight,
          8 => example_8_load_model,
          9 => example_9_product_pricing,
          10 => example_10_cross_validation,
          11 => example_11_text_processing,
          12 => example_12_classification_metrics
        }
        examples[num]
      end

      def self.example_1_basic_training
        <<~RUBY
          require 'grydra'
          
          # Training data
          training_data = [
            { name: "Company 1", num_employees: 5, is_new: false, site: true, label: 0 },
            { name: "Company 2", num_employees: 4, is_new: true, site: false, label: 0 },
            { name: "Company 3", num_employees: 4, is_new: false, site: false, label: 1 },
            { name: "Company 4", num_employees: 20, is_new: false, site: false, label: 1 },
            { name: "Company 5", num_employees: 60, is_new: false, site: false, label: 1 }
          ]
          
          # Create and train model
          model = GRYDRA::Networks::EasyNetwork.new(true, true)
          model.configure_adam_optimizer(0.001, 0.9, 0.999)
          
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
          
          # Save model
          GRYDRA::Utils::Persistence.save_model(model, "company_model")
          puts "✅ Training completed!"
        RUBY
      end

      def self.example_2_load_and_predict
        <<~RUBY
          require 'grydra'
          
          # Load model
          model = GRYDRA::Utils::Persistence.load_model("company_model")
          
          # New data
          new_data = [
            { name: "New Company A", num_employees: 12, is_new: true, site: true },
            { name: "New Company B", num_employees: 50, is_new: false, site: false }
          ]
          
          # Predict
          predictions = model.predict_hashes(new_data, [:num_employees, :is_new, :site])
          
          new_data.each_with_index do |company, i|
            prediction = predictions[i].first.round(3)
            label = prediction >= 0.5 ? 'Label 1 (Yes)' : 'Label 0 (No)'
            puts "Company: \#{company[:name]} → Prediction: \#{prediction} (\#{label})"
          end
        RUBY
      end

      def self.example_3_xor_problem
        <<~RUBY
          require 'grydra'
          
          # XOR problem
          network = GRYDRA::Networks::MainNetwork.new(true)
          network.add_subnet([2, 3, 1], [:tanh, :tanh])
          network.add_subnet([2, 4, 1], [:sigmoid, :sigmoid])
          
          inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
          outputs = [[0], [1], [1], [0]]
          
          network.train_subnets(
            [
              {input: inputs, output: outputs},
              {input: inputs, output: outputs}
            ],
            0.9,
            6000,
            batch_size: 1,
            patience: 100,
            decay: 0.995
          )
          
          puts "\\n📊 XOR Evaluation:"
          inputs.each do |input|
            output = network.combine_results(input)
            puts "Input: \#{input} => Output: \#{output.map { |v| v.round(3) }}"
          end
        RUBY
      end

      # Add more examples as needed...
      def self.example_4_temperature_conversion
        "# Temperature conversion example - See documentation"
      end

      def self.example_5_zscore_normalization
        "# Z-score normalization example - See documentation"
      end

      def self.example_6_weight_prediction
        "# Weight prediction example - See documentation"
      end

      def self.example_7_advanced_weight
        "# Advanced weight prediction - See documentation"
      end

      def self.example_8_load_model
        "# Load and use saved model - See documentation"
      end

      def self.example_9_product_pricing
        "# Product pricing example - See documentation"
      end

      def self.example_10_cross_validation
        "# Cross-validation example - See documentation"
      end

      def self.example_11_text_processing
        "# Text processing with TF-IDF - See documentation"
      end

      def self.example_12_classification_metrics
        "# Classification metrics example - See documentation"
      end
    end
  end
end
