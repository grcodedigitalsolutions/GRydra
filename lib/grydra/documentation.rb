module GRYDRA
  module Documentation
    METHOD_DESCRIPTIONS = {
      # Networks
      'Networks::MainNetwork.add_subnet' => {
        description: 'Adds a subnet to the main network with the given structure.',
        example: 'network = GRYDRA::Networks::MainNetwork.new; network.add_subnet([2, 4, 1])'
      },
      'Networks::EasyNetwork.train_numerical' => {
        description: 'Trains the network with numerical data.',
        example: 'network.train_numerical(data_input, data_output, [[4, 1]], 0.05, 15000, :max)'
      },
      'Networks::EasyNetwork.configure_adam_optimizer' => {
        description: 'Configures the Adam optimizer for all subnets.',
        example: 'network.configure_adam_optimizer(0.001, 0.9, 0.999)'
      },
      
      # Metrics
      'Metrics.mse' => {
        description: 'Calculates Mean Squared Error.',
        example: 'GRYDRA::Metrics.mse(predictions, actuals)'
      },
      'Metrics.accuracy' => {
        description: 'Calculates classification accuracy.',
        example: 'GRYDRA::Metrics.accuracy(predictions, actuals, 0.5)'
      },
      
      # Persistence
      'Utils::Persistence.save_model' => {
        description: 'Saves the trained model to a binary file.',
        example: 'GRYDRA::Utils::Persistence.save_model(model, "my_model", "./models")'
      },
      'Utils::Persistence.load_model' => {
        description: 'Loads a saved model from a binary file.',
        example: 'model = GRYDRA::Utils::Persistence.load_model("my_model", "./models")'
      },
      
      # Visualization
      'Utils::Visualization.plot_architecture_ascii' => {
        description: 'Displays an ASCII representation of the network architecture.',
        example: 'GRYDRA::Utils::Visualization.plot_architecture_ascii(model)'
      },
      'Utils::Visualization.analyze_gradients' => {
        description: 'Analyzes the model\'s gradients.',
        example: 'analysis = GRYDRA::Utils::Visualization.analyze_gradients(model)'
      },
      
      # Training
      'Training::CrossValidation.cross_validation' => {
        description: 'Performs k-fold cross-validation.',
        example: 'result = GRYDRA::Training::CrossValidation.cross_validation(data_x, data_y, 5) { |train_x, train_y, test_x, test_y| ... }'
      },
      'Training::HyperparameterSearch.hyperparameter_search' => {
        description: 'Performs hyperparameter search using grid search.',
        example: 'result = GRYDRA::Training::HyperparameterSearch.hyperparameter_search(data_x, data_y, param_grid) { |params, x, y| ... }'
      }
    }

    def self.describe_method(class_name, method_name)
      key = "#{class_name}.#{method_name}"
      info = METHOD_DESCRIPTIONS[key]

      if info
        puts "\e[1;36m📖 Description of #{key}:\e[0m"
        puts info[:description]
        puts "\n💡 Example:"
        puts "#{info[:example]}\e[0m"
      else
        puts "\e[31;1m✗ No description found for method '#{key}'\e[0m"
        puts "\e[36m💡 You can call: GRYDRA::Documentation.list_methods_available\e[0m"
      end
    end

    def self.list_methods_available
      puts "\e[1;36m📚 Documented public methods:\e[0m"
      grouped = METHOD_DESCRIPTIONS.keys.group_by { |k| k.split('.').first }
      grouped.each do |class_name, methods|
        puts "  #{class_name}:"
        methods.each { |m| puts "    - #{m.split('.').last}" }
      end
      print "\e[0m"
    end
  end
end
