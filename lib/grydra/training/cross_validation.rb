module GRYDRA
  module Training
    module CrossValidation
      # K-fold cross-validation
      def self.cross_validation(data_input, data_output, k_folds = 5)
        indices = (0...data_input.size).to_a.shuffle
        fold_size = data_input.size / k_folds
        errors = []

        k_folds.times do |i|
          start = i * fold_size
          finish = [start + fold_size, data_input.size].min
          indices_test = indices[start...finish]
          indices_train = indices - indices_test

          # Split data
          train_x = indices_train.map { |idx| data_input[idx] }
          train_y = indices_train.map { |idx| data_output[idx] }
          test_x = indices_test.map { |idx| data_input[idx] }
          test_y = indices_test.map { |idx| data_output[idx] }

          # Train and evaluate
          error = yield(train_x, train_y, test_x, test_y)
          errors << error
        end

        {
          errors: errors,
          average: errors.sum / errors.size.to_f,
          deviation: Math.sqrt(errors.map { |e| (e - errors.sum / errors.size.to_f)**2 }.sum / errors.size.to_f)
        }
      end
    end
  end
end
