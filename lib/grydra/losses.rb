module GRYDRA
  module Losses
    # Mean Squared Error Loss
    def self.mse(predictions, targets)
      n = predictions.size
      sum = predictions.zip(targets).map { |p, t| (p - t)**2 }.sum
      sum / n.to_f
    end

    def self.mse_derivative(predictions, targets)
      predictions.zip(targets).map { |p, t| 2 * (p - t) / predictions.size }
    end

    # Mean Absolute Error Loss
    def self.mae(predictions, targets)
      n = predictions.size
      sum = predictions.zip(targets).map { |p, t| (p - t).abs }.sum
      sum / n.to_f
    end

    def self.mae_derivative(predictions, targets)
      predictions.zip(targets).map { |p, t| p > t ? 1.0 / predictions.size : -1.0 / predictions.size }
    end

    # Binary Cross-Entropy Loss
    def self.binary_crossentropy(predictions, targets, epsilon: 1e-7)
      predictions = predictions.map { |p| [[p, epsilon].max, 1 - epsilon].min }
      n = predictions.size
      sum = predictions.zip(targets).map do |p, t|
        -(t * Math.log(p) + (1 - t) * Math.log(1 - p))
      end.sum
      sum / n.to_f
    end

    def self.binary_crossentropy_derivative(predictions, targets, epsilon: 1e-7)
      predictions = predictions.map { |p| [[p, epsilon].max, 1 - epsilon].min }
      predictions.zip(targets).map do |p, t|
        (p - t) / (p * (1 - p) * predictions.size)
      end
    end

    # Categorical Cross-Entropy Loss
    def self.categorical_crossentropy(predictions, targets, epsilon: 1e-7)
      predictions = predictions.map { |p| [p, epsilon].max }
      n = predictions.size
      sum = predictions.zip(targets).map do |pred_row, target_row|
        pred_row.zip(target_row).map { |p, t| -t * Math.log(p) }.sum
      end.sum
      sum / n.to_f
    end

    # Huber Loss (robust to outliers)
    def self.huber(predictions, targets, delta: 1.0)
      n = predictions.size
      sum = predictions.zip(targets).map do |p, t|
        error = (p - t).abs
        if error <= delta
          0.5 * error**2
        else
          delta * (error - 0.5 * delta)
        end
      end.sum
      sum / n.to_f
    end

    def self.huber_derivative(predictions, targets, delta: 1.0)
      predictions.zip(targets).map do |p, t|
        error = p - t
        if error.abs <= delta
          error / predictions.size
        else
          delta * (error > 0 ? 1 : -1) / predictions.size
        end
      end
    end

    # Hinge Loss (for SVM-style classification)
    def self.hinge(predictions, targets)
      n = predictions.size
      sum = predictions.zip(targets).map do |p, t|
        [0, 1 - t * p].max
      end.sum
      sum / n.to_f
    end

    def self.hinge_derivative(predictions, targets)
      predictions.zip(targets).map do |p, t|
        (1 - t * p) > 0 ? -t / predictions.size : 0
      end
    end

    # Log-Cosh Loss (smooth approximation of MAE)
    def self.log_cosh(predictions, targets)
      n = predictions.size
      sum = predictions.zip(targets).map do |p, t|
        x = p - t
        Math.log(Math.cosh(x))
      end.sum
      sum / n.to_f
    end

    def self.log_cosh_derivative(predictions, targets)
      predictions.zip(targets).map do |p, t|
        x = p - t
        Math.tanh(x) / predictions.size
      end
    end

    # Quantile Loss (for quantile regression)
    def self.quantile(predictions, targets, quantile: 0.5)
      n = predictions.size
      sum = predictions.zip(targets).map do |p, t|
        error = t - p
        error > 0 ? quantile * error : (quantile - 1) * error
      end.sum
      sum / n.to_f
    end
  end
end
