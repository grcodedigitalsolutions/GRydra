module GRYDRA
  module Normalization
    # Z-score normalization
    def self.zscore_normalize(data)
      n = data.size
      means = data.first.size.times.map { |i| data.map { |row| row[i] }.sum.to_f / n }
      std_devs = data.first.size.times.map do |i|
        m = means[i]
        Math.sqrt(data.map { |row| (row[i] - m)**2 }.sum.to_f / n)
      end
      normalized = data.map do |row|
        row.each_with_index.map { |value, i| std_devs[i] != 0 ? (value - means[i]) / std_devs[i] : 0 }
      end
      [normalized, means, std_devs]
    end

    def self.zscore_denormalize(normalized, means, std_devs)
      normalized.map do |row|
        row.each_with_index.map { |value, i| value * std_devs[i] + means[i] }
      end
    end

    # Min-Max normalization
    def self.min_max_normalize(data, min_val = 0, max_val = 1)
      data_min = data.flatten.min
      data_max = data.flatten.max
      range = data_max - data_min
      return data if range == 0

      data.map do |row|
        row.map { |v| min_val + (v - data_min) * (max_val - min_val) / range }
      end
    end

    # Generic normalization with multiple methods
    def self.normalize_multiple(data, max_values, method = :max)
      case method
      when :max
        data.map do |row|
          row.each_with_index.map { |value, idx| value.to_f / max_values[idx] }
        end
      when :zscore
        means = max_values[:means]
        std_devs = max_values[:std_devs]
        data.map do |row|
          row.each_with_index.map do |value, idx|
            std_devs[idx] != 0 ? (value.to_f - means[idx]) / std_devs[idx] : 0
          end
        end
      else
        raise ArgumentError, "Unknown normalization method: #{method}"
      end
    end

    def self.calculate_max_values(data, method = :max)
      if method == :max
        max_values = {}
        data.first.size.times do |i|
          max_values[i] = data.map { |row| row[i] }.max.to_f
        end
        max_values
      elsif method == :zscore
        n = data.size
        means = data.first.size.times.map do |i|
          data.map { |row| row[i] }.sum.to_f / n
        end
        std_devs = data.first.size.times.map do |i|
          m = means[i]
          Math.sqrt(data.map { |row| (row[i] - m)**2 }.sum.to_f / n)
        end
        { means: means, std_devs: std_devs }
      else
        raise ArgumentError, "Unknown method for calculating max values: #{method}"
      end
    end
  end
end
