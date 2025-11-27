module GRYDRA
  module Activations
    # Hyperbolic tangent activation function
    def self.tanh(x)
      Math.tanh(x)
    end

    def self.derivative_tanh(x)
      1 - tanh(x)**2
    end

    # Rectified Linear Unit
    def self.relu(x)
      x > 0 ? x : 0
    end

    def self.derivative_relu(x)
      x > 0 ? 1 : 0
    end

    # Sigmoid activation function
    def self.sigmoid(x)
      1.0 / (1.0 + Math.exp(-x))
    end

    def self.derivative_sigmoid(x)
      s = sigmoid(x)
      s * (1 - s)
    end

    # Softmax for multi-class classification
    def self.softmax(vector)
      max = vector.max
      exps = vector.map { |v| Math.exp(v - max) }
      sum = exps.sum
      exps.map { |v| v / sum }
    end

    # Leaky ReLU
    def self.leaky_relu(x, alpha = 0.01)
      x > 0 ? x : alpha * x
    end

    def self.derivative_leaky_relu(x, alpha = 0.01)
      x > 0 ? 1 : alpha
    end

    # Swish activation function
    def self.swish(x)
      x * sigmoid(x)
    end

    def self.derivative_swish(x)
      s = sigmoid(x)
      s + x * s * (1 - s)
    end

    # GELU (Gaussian Error Linear Unit)
    def self.gelu(x)
      0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math::PI) * (x + 0.044715 * x**3)))
    end

    def self.derivative_gelu(x)
      tanh_arg = Math.sqrt(2 / Math::PI) * (x + 0.044715 * x**3)
      tanh_val = Math.tanh(tanh_arg)
      sech2 = 1 - tanh_val**2
      0.5 * (1 + tanh_val) + 0.5 * x * sech2 * Math.sqrt(2 / Math::PI) * (1 + 3 * 0.044715 * x**2)
    end
  end
end
