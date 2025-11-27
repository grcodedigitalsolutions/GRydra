module GRYDRA
  module Callbacks
    # Base Callback class
    class Base
      def on_train_begin(logs = {}); end
      def on_train_end(logs = {}); end
      def on_epoch_begin(epoch, logs = {}); end
      def on_epoch_end(epoch, logs = {}); end
      def on_batch_begin(batch, logs = {}); end
      def on_batch_end(batch, logs = {}); end
    end

    # Early Stopping Callback
    class EarlyStopping < Base
      attr_reader :stopped_epoch

      def initialize(monitor: :loss, patience: 10, min_delta: 0.0, mode: :min, restore_best: true)
        @monitor = monitor
        @patience = patience
        @min_delta = min_delta
        @mode = mode
        @restore_best = restore_best
        @wait = 0
        @stopped_epoch = 0
        @best_value = mode == :min ? Float::INFINITY : -Float::INFINITY
        @best_weights = nil
      end

      def on_epoch_end(epoch, logs = {})
        current = logs[@monitor]
        return unless current

        if improved?(current)
          @best_value = current
          @wait = 0
          @best_weights = logs[:weights].dup if @restore_best && logs[:weights]
        else
          @wait += 1
          if @wait >= @patience
            @stopped_epoch = epoch
            logs[:stop_training] = true
            logs[:weights] = @best_weights if @restore_best && @best_weights
            puts "Early stopping at epoch #{epoch + 1}"
          end
        end
      end

      private

      def improved?(current)
        if @mode == :min
          current < @best_value - @min_delta
        else
          current > @best_value + @min_delta
        end
      end
    end

    # Learning Rate Scheduler
    class LearningRateScheduler < Base
      def initialize(schedule: nil, &block)
        @schedule = schedule || block
      end

      def on_epoch_begin(epoch, logs = {})
        new_lr = @schedule.call(epoch, logs[:learning_rate])
        logs[:learning_rate] = new_lr if new_lr
      end
    end

    # Reduce Learning Rate on Plateau
    class ReduceLROnPlateau < Base
      def initialize(monitor: :loss, factor: 0.5, patience: 5, min_lr: 1e-7, mode: :min)
        @monitor = monitor
        @factor = factor
        @patience = patience
        @min_lr = min_lr
        @mode = mode
        @wait = 0
        @best_value = mode == :min ? Float::INFINITY : -Float::INFINITY
      end

      def on_epoch_end(epoch, logs = {})
        current = logs[@monitor]
        return unless current

        if improved?(current)
          @best_value = current
          @wait = 0
        else
          @wait += 1
          if @wait >= @patience
            old_lr = logs[:learning_rate]
            new_lr = [old_lr * @factor, @min_lr].max
            if new_lr < old_lr
              logs[:learning_rate] = new_lr
              puts "Reducing learning rate to #{new_lr.round(8)}"
            end
            @wait = 0
          end
        end
      end

      private

      def improved?(current)
        if @mode == :min
          current < @best_value
        else
          current > @best_value
        end
      end
    end

    # Model Checkpoint Callback
    class ModelCheckpoint < Base
      def initialize(filepath, monitor: :loss, save_best_only: true, mode: :min)
        @filepath = filepath
        @monitor = monitor
        @save_best_only = save_best_only
        @mode = mode
        @best_value = mode == :min ? Float::INFINITY : -Float::INFINITY
      end

      def on_epoch_end(epoch, logs = {})
        current = logs[@monitor]
        
        if @save_best_only
          if improved?(current)
            @best_value = current
            save_model(logs[:model], epoch)
          end
        else
          save_model(logs[:model], epoch)
        end
      end

      private

      def improved?(current)
        if @mode == :min
          current < @best_value
        else
          current > @best_value
        end
      end

      def save_model(model, epoch)
        filepath = @filepath.gsub('{epoch}', epoch.to_s)
        Utils::Persistence.save_model(model, filepath)
      end
    end

    # CSV Logger Callback
    class CSVLogger < Base
      def initialize(filename, separator: ',', append: false)
        @filename = filename
        @separator = separator
        @append = append
        @file = nil
        @keys = nil
      end

      def on_train_begin(logs = {})
        mode = @append ? 'a' : 'w'
        @file = File.open(@filename, mode)
      end

      def on_epoch_end(epoch, logs = {})
        if @keys.nil?
          @keys = ['epoch'] + logs.keys.map(&:to_s).sort
          @file.puts @keys.join(@separator) unless @append
        end

        values = [epoch] + @keys[1..-1].map { |k| logs[k.to_sym] || '' }
        @file.puts values.join(@separator)
        @file.flush
      end

      def on_train_end(logs = {})
        @file.close if @file
      end
    end

    # Progress Bar Callback
    class ProgressBar < Base
      def initialize(total_epochs)
        @total_epochs = total_epochs
        @start_time = nil
      end

      def on_train_begin(logs = {})
        @start_time = Time.now
      end

      def on_epoch_end(epoch, logs = {})
        progress = (epoch + 1).to_f / @total_epochs
        bar_length = 30
        filled = (bar_length * progress).to_i
        bar = '=' * filled + '-' * (bar_length - filled)
        
        elapsed = Time.now - @start_time
        eta = elapsed / progress - elapsed
        
        metrics = logs.map { |k, v| "#{k}: #{v.is_a?(Numeric) ? v.round(6) : v}" }.join(', ')
        
        print "\rEpoch #{epoch + 1}/#{@total_epochs} [#{bar}] - #{elapsed.round(1)}s - ETA: #{eta.round(1)}s - #{metrics}"
        puts if epoch + 1 == @total_epochs
      end
    end
  end
end
