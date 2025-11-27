module GRYDRA
  module Layers
    # Base Layer class
    class Base
      def calculate_outputs(inputs)
        raise NotImplementedError, 'Subclasses must implement calculate_outputs method'
      end
    end
  end
end
