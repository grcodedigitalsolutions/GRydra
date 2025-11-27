module GRYDRA
  module Initializers
    # Xavier/Glorot initialization
    def self.xavier_init(num_inputs)
      limit = Math.sqrt(6.0 / num_inputs)
      rand * 2 * limit - limit
    end

    # He initialization (for ReLU)
    def self.he_init(num_inputs)
      Math.sqrt(2.0 / num_inputs) * (rand * 2 - 1)
    end
  end
end
