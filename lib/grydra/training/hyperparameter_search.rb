module GRYDRA
  module Training
    module HyperparameterSearch
      # Grid search for hyperparameters
      def self.hyperparameter_search(data_x, data_y, param_grid, verbose: true)
        best_params = nil
        best_score = Float::INFINITY
        results = []

        puts 'Starting hyperparameter search...' if verbose

        param_grid.each_with_index do |params, idx|
          puts "Testing configuration #{idx + 1}/#{param_grid.size}: #{params}" if verbose

          begin
            score = yield(params, data_x, data_y)
            results << { parameters: params, score: score }

            if score < best_score
              best_score = score
              best_params = params
              puts "  New best configuration! Score: #{score.round(6)}" if verbose
            else
              puts "  Score: #{score.round(6)}" if verbose
            end
          rescue StandardError => e
            puts "  Error with this configuration: #{e.message}" if verbose
            results << { parameters: params, score: Float::INFINITY, error: e.message }
          end
        end

        if verbose
          puts "\nBest parameters found:"
          puts "  Configuration: #{best_params}"
          puts "  Score: #{best_score.round(6)}"
        end

        {
          parameters: best_params,
          score: best_score,
          all_results: results.sort_by { |r| r[:score] }
        }
      end
    end
  end
end
