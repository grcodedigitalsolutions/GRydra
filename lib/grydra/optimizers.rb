module GRYDRA
  module Optimizers
    # Base Optimizer class
    class Base
      def update(parameter_id, gradient)
        raise NotImplementedError, 'Subclasses must implement update method'
      end

      def reset
        raise NotImplementedError, 'Subclasses must implement reset method'
      end
    end

    # Stochastic Gradient Descent with Momentum
    class SGD < Base
      attr_reader :learning_rate, :momentum

      def initialize(learning_rate: 0.01, momentum: 0.9, nesterov: false)
        @learning_rate = learning_rate
        @momentum = momentum
        @nesterov = nesterov
        @velocity = {}
      end

      def update(parameter_id, gradient)
        @velocity[parameter_id] ||= 0
        
        if @nesterov
          # Nesterov momentum
          @velocity[parameter_id] = @momentum * @velocity[parameter_id] - @learning_rate * gradient
          @momentum * @velocity[parameter_id] - @learning_rate * gradient
        else
          # Classical momentum
          @velocity[parameter_id] = @momentum * @velocity[parameter_id] - @learning_rate * gradient
          @velocity[parameter_id]
        end
      end

      def reset
        @velocity.clear
      end
    end

    # Adam Optimizer (Adaptive Moment Estimation)
    class Adam < Base
      attr_reader :alpha, :beta1, :beta2, :epsilon

      def initialize(alpha: 0.001, beta1: 0.9, beta2: 0.999, epsilon: 1e-8)
        @alpha = alpha
        @beta1 = beta1
        @beta2 = beta2
        @epsilon = epsilon
        @m = {}
        @v = {}
        @t = 0
      end

      def update(parameter_id, gradient)
        @t += 1
        @m[parameter_id] ||= 0
        @v[parameter_id] ||= 0

        @m[parameter_id] = @beta1 * @m[parameter_id] + (1 - @beta1) * gradient
        @v[parameter_id] = @beta2 * @v[parameter_id] + (1 - @beta2) * gradient**2

        m_hat = @m[parameter_id] / (1 - @beta1**@t)
        v_hat = @v[parameter_id] / (1 - @beta2**@t)

        @alpha * m_hat / (Math.sqrt(v_hat) + @epsilon)
      end

      def reset
        @m.clear
        @v.clear
        @t = 0
      end
    end

    # RMSprop Optimizer
    class RMSprop < Base
      attr_reader :learning_rate, :decay_rate, :epsilon

      def initialize(learning_rate: 0.001, decay_rate: 0.9, epsilon: 1e-8)
        @learning_rate = learning_rate
        @decay_rate = decay_rate
        @epsilon = epsilon
        @cache = {}
      end

      def update(parameter_id, gradient)
        @cache[parameter_id] ||= 0
        @cache[parameter_id] = @decay_rate * @cache[parameter_id] + (1 - @decay_rate) * gradient**2
        @learning_rate * gradient / (Math.sqrt(@cache[parameter_id]) + @epsilon)
      end

      def reset
        @cache.clear
      end
    end

    # AdaGrad Optimizer
    class AdaGrad < Base
      attr_reader :learning_rate, :epsilon

      def initialize(learning_rate: 0.01, epsilon: 1e-8)
        @learning_rate = learning_rate
        @epsilon = epsilon
        @cache = {}
      end

      def update(parameter_id, gradient)
        @cache[parameter_id] ||= 0
        @cache[parameter_id] += gradient**2
        @learning_rate * gradient / (Math.sqrt(@cache[parameter_id]) + @epsilon)
      end

      def reset
        @cache.clear
      end
    end

    # AdamW Optimizer (Adam with decoupled weight decay)
    class AdamW < Base
      attr_reader :alpha, :beta1, :beta2, :epsilon, :weight_decay

      def initialize(alpha: 0.001, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, weight_decay: 0.01)
        @alpha = alpha
        @beta1 = beta1
        @beta2 = beta2
        @epsilon = epsilon
        @weight_decay = weight_decay
        @m = {}
        @v = {}
        @t = 0
      end

      def update(parameter_id, gradient, parameter_value: 0)
        @t += 1
        @m[parameter_id] ||= 0
        @v[parameter_id] ||= 0

        @m[parameter_id] = @beta1 * @m[parameter_id] + (1 - @beta1) * gradient
        @v[parameter_id] = @beta2 * @v[parameter_id] + (1 - @beta2) * gradient**2

        m_hat = @m[parameter_id] / (1 - @beta1**@t)
        v_hat = @v[parameter_id] / (1 - @beta2**@t)

        # AdamW: decoupled weight decay
        @alpha * (m_hat / (Math.sqrt(v_hat) + @epsilon) + @weight_decay * parameter_value)
      end

      def reset
        @m.clear
        @v.clear
        @t = 0
      end
    end

    # Alias for backward compatibility
    AdamOptimizer = Adam
  end
end
