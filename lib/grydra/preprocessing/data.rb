module GRYDRA
  module Preprocessing
    module Data
      # Split data into training and test sets
      def self.split_data(data_x, data_y, training_ratio = 0.8, seed = nil)
        srand(seed) if seed
        indices = (0...data_x.size).to_a.shuffle
        cut = (data_x.size * training_ratio).to_i

        {
          train_x: indices[0...cut].map { |i| data_x[i] },
          train_y: indices[0...cut].map { |i| data_y[i] },
          test_x: indices[cut..-1].map { |i| data_x[i] },
          test_y: indices[cut..-1].map { |i| data_y[i] }
        }
      end

      # Generate synthetic data for testing
      def self.generate_synthetic_data(n_samples, n_features, noise = 0.1, seed = nil)
        srand(seed) if seed
        data = Array.new(n_samples) do
          Array.new(n_features) { rand * 2 - 1 + (rand * noise - noise / 2) }
        end

        # Generate labels based on a simple function
        labels = data.map do |sample|
          value = sample.each_with_index.sum { |x, i| x * (i + 1) * 0.1 }
          [value + (rand * noise - noise / 2)]
        end

        { data: data, labels: labels }
      end

      # Convert hashes to vectors
      def self.convert_hashes_to_vectors(array_hashes, keys)
        array_hashes.map do |hash|
          keys.map do |k|
            if hash[k]
              hash[k] == true ? 1.0 : (hash[k] == false ? 0.0 : hash[k].to_f)
            else
              0.0
            end
          end
        end
      end
    end
  end
end
