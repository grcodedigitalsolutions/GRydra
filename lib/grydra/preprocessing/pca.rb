module GRYDRA
  module Preprocessing
    module PCA
      # Principal Component Analysis with Power Iteration
      def self.pca(data, components: 2, max_iterations: 1000, tolerance: 1e-6)
        n = data.size
        m = data.first.size

        # Center data
        means = (0...m).map { |i| data.map { |row| row[i] }.sum.to_f / n }
        centered_data = data.map { |row| row.zip(means).map { |v, mean| v - mean } }

        # Calculate covariance matrix
        covariance = calculate_covariance(centered_data, n, m)

        # Find principal components using power iteration
        principal_components = []
        eigenvalues = []
        
        components.times do |comp_idx|
          eigenvector, eigenvalue = power_iteration(
            covariance, 
            max_iterations: max_iterations, 
            tolerance: tolerance,
            deflate: principal_components
          )
          
          break unless eigenvector
          
          principal_components << eigenvector
          eigenvalues << eigenvalue
        end

        # Project data onto principal components
        transformed_data = centered_data.map do |row|
          principal_components.map do |pc|
            row.zip(pc).map { |a, b| a * b }.sum
          end
        end

        {
          means: means,
          principal_components: principal_components,
          eigenvalues: eigenvalues,
          explained_variance: calculate_explained_variance(eigenvalues),
          transformed_data: transformed_data,
          covariance: covariance
        }
      end

      def self.transform(data, pca_result)
        # Transform new data using existing PCA
        centered = data.map do |row|
          row.zip(pca_result[:means]).map { |v, mean| v - mean }
        end

        centered.map do |row|
          pca_result[:principal_components].map do |pc|
            row.zip(pc).map { |a, b| a * b }.sum
          end
        end
      end

      private

      def self.calculate_covariance(centered_data, n, m)
        covariance = Array.new(m) { Array.new(m, 0.0) }
        (0...m).each do |i|
          (0...m).each do |j|
            covariance[i][j] = centered_data.map { |row| row[i] * row[j] }.sum / (n - 1).to_f
          end
        end
        covariance
      end

      def self.power_iteration(matrix, max_iterations:, tolerance:, deflate: [])
        n = matrix.size
        
        # Initialize random vector
        vector = Array.new(n) { rand }
        vector = normalize_vector(vector)

        # Deflate for previously found components
        deflate.each do |pc|
          projection = vector.zip(pc).map { |a, b| a * b }.sum
          vector = vector.zip(pc).map { |v, p| v - projection * p }
        end
        vector = normalize_vector(vector)

        max_iterations.times do
          # Multiply matrix by vector
          new_vector = matrix_vector_multiply(matrix, vector)
          
          # Normalize
          new_vector = normalize_vector(new_vector)
          
          # Check convergence
          diff = vector.zip(new_vector).map { |a, b| (a - b).abs }.max
          
          vector = new_vector
          
          break if diff < tolerance
        end

        # Calculate eigenvalue (Rayleigh quotient)
        numerator = matrix_vector_multiply(matrix, vector).zip(vector).map { |a, b| a * b }.sum
        denominator = vector.map { |v| v**2 }.sum
        eigenvalue = numerator / denominator

        [vector, eigenvalue]
      end

      def self.matrix_vector_multiply(matrix, vector)
        matrix.map do |row|
          row.zip(vector).map { |a, b| a * b }.sum
        end
      end

      def self.normalize_vector(vector)
        magnitude = Math.sqrt(vector.map { |v| v**2 }.sum)
        return vector if magnitude.zero?
        vector.map { |v| v / magnitude }
      end

      def self.calculate_explained_variance(eigenvalues)
        total = eigenvalues.sum
        return [] if total.zero?
        eigenvalues.map { |ev| ev / total }
      end
    end
  end
end
