module GRYDRA
  module Networks
    # Easy Network - Simplified interface for quick training
    class EasyNetwork
      attr_accessor :network, :vocabulary, :max_values, :max_values_output

      def initialize(print_epochs: false, plot: false, verbose: true)
        @network = MainNetwork.new(print_epochs: print_epochs, plot: plot)
        @vocabulary = nil
        @max_values = {}
        @max_values_output = {}
        @verbose = verbose
      end

      def configure_adam_optimizer(alpha = 0.001, beta1 = 0.9, beta2 = 0.999)
        @network.subnets.each { |subnet| subnet.use_adam_optimizer(alpha, beta1, beta2) }
      end

      def evaluate_model(data_test_x, data_test_y, metrics = %i[mse mae], normalization = :max)
        predictions = predict_numerical(data_test_x, normalization)
        results = {}
        
        # Handle both single and multi-output cases
        predictions_flat = predictions.flatten
        actuals_flat = data_test_y.flatten
        
        # Ensure we have valid data
        return results if predictions_flat.empty? || actuals_flat.empty?
        return results if predictions_flat.any?(&:nil?) || actuals_flat.any?(&:nil?)

        metrics.each do |metric|
          case metric
          when :mse
            results[:mse] = Metrics.mse(predictions_flat, actuals_flat)
          when :mae
            results[:mae] = Metrics.mae(predictions_flat, actuals_flat)
          when :accuracy
            results[:accuracy] = Metrics.accuracy(predictions_flat, actuals_flat)
          when :confusion_matrix
            results[:confusion_matrix] = Metrics.confusion_matrix(predictions_flat, actuals_flat)
          when :auc_roc
            results[:auc_roc] = Metrics.auc_roc(predictions_flat, actuals_flat)
          end
        end
        results
      end

      # Train with hash data
      def train_hashes(data_hash, input_keys, label_key, structures, rate, epochs, normalization = :max, **opts)
        @network.subnets.clear

        inputs = data_hash.map do |item|
          input_keys.map do |key|
            value = item[key]
            value == true ? 1.0 : (value == false ? 0.0 : value.to_f)
          end
        end

        @max_values = Normalization.calculate_max_values(inputs, normalization)
        data_normalized = Normalization.normalize_multiple(inputs, @max_values, normalization)

        labels = data_hash.map { |item| [item[label_key].to_f] }
        @max_values_output = Normalization.calculate_max_values(labels, normalization)
        labels_no = Normalization.normalize_multiple(labels, @max_values_output, normalization)

        structures.each do |structure|
          @network.add_subnet([input_keys.size, *structure])
        end

        data_for_subnets = structures.map { |_| { input: data_normalized, output: labels_no } }
        @network.train_subnets(data_for_subnets, rate, epochs, **opts)
      end

      def predict_hashes(new_hashes, input_keys, normalization = :max)
        inputs = new_hashes.map do |item|
          input_keys.map do |key|
            value = item[key]
            value == true ? 1.0 : (value == false ? 0.0 : value.to_f)
          end
        end

        data_normalized = Normalization.normalize_multiple(inputs, @max_values, normalization)
        denormalize_predictions(data_normalized, normalization)
      end

      # Train with numerical data
      def train_numerical(data_input, data_output, structures, rate, epochs, normalization = :max, **opts)
        @network.subnets.clear

        @max_values = Normalization.calculate_max_values(data_input, normalization)
        @max_values_output = Normalization.calculate_max_values(data_output, normalization)

        data_input_no = Normalization.normalize_multiple(data_input, @max_values, normalization)
        data_output_no = Normalization.normalize_multiple(data_output, @max_values_output, normalization)

        structures.each do |structure|
          @network.add_subnet([data_input.first.size, *structure])
        end

        data_for_subnets = structures.map { |_| { input: data_input_no, output: data_output_no } }
        @network.train_subnets(data_for_subnets, rate, epochs, **opts)
      end

      def predict_numerical(new_data, normalization = :max)
        data_normalized = Normalization.normalize_multiple(new_data, @max_values, normalization)
        denormalize_predictions(data_normalized, normalization)
      end

      # Train with text data
      def train_text(texts, labels, structures, rate, epochs, normalization = :max, **opts)
        @network.subnets.clear
        @vocabulary = Preprocessing::Text.create_vocabulary(texts)

        inputs = texts.map { |text| Preprocessing::Text.vectorize_text(text, @vocabulary) }
        
        # For text, use simple max normalization
        max_val = @vocabulary.size.to_f
        data_normalized = inputs.map { |input| input.map { |v| v / max_val } }
        @max_values = { 0 => max_val }

        @max_values_output = Normalization.calculate_max_values(labels, normalization)
        labels_no = Normalization.normalize_multiple(labels, @max_values_output, normalization)

        structures.each do |structure|
          @network.add_subnet([@vocabulary.size, *structure])
        end

        data_for_subnets = structures.map { |_| { input: data_normalized, output: labels_no } }
        @network.train_subnets(data_for_subnets, rate, epochs, **opts)
      end

      def predict_text(new_texts, normalization = :max)
        inputs = new_texts.map { |text| Preprocessing::Text.vectorize_text(text, @vocabulary) }
        
        # For text, we use a simple max normalization based on vocabulary size
        max_val = @vocabulary.size.to_f
        data_normalized = inputs.map { |input| input.map { |v| v / max_val } }
        
        denormalize_predictions(data_normalized, normalization)
      end

      private

      def denormalize_predictions(data_normalized, normalization)
        data_normalized.map do |input|
          pred_norm = @network.combine_results(input)
          if normalization == :zscore && @max_values_output.is_a?(Hash) && @max_values_output.key?(:std_devs)
            pred_norm.map.with_index do |val, idx|
              val * @max_values_output[:std_devs][idx] + @max_values_output[:means][idx]
            end
          else
            pred_norm.map.with_index do |val, idx|
              max_val = @max_values_output[idx] || @max_values_output.values.first || 1.0
              val * max_val
            end
          end
        end
      end
    end
  end
end
