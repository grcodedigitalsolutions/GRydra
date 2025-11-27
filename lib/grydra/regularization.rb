module GRYDRA
  module Regularization
    # Apply dropout to outputs during training
    def self.apply_dropout(outputs, dropout_rate = 0.5, training = true)
      return outputs unless training
      outputs.map { |s| rand < dropout_rate ? 0 : s / (1 - dropout_rate) }
    end

    # L1 regularization (Lasso)
    def self.l1_regularization(weights, lambda_l1)
      lambda_l1 * weights.sum { |p| p.abs }
    end

    # L2 regularization (Ridge)
    def self.l2_regularization(weights, lambda_l2)
      lambda_l2 * weights.sum { |p| p**2 }
    end
  end
end
