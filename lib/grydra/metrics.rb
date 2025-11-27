module GRYDRA
  module Metrics
    # Mean Squared Error
    def self.mse(predictions, actuals)
      n = predictions.size
      sum = predictions.zip(actuals).map { |p, r| (p - r)**2 }.sum
      sum / n.to_f
    end

    # Mean Absolute Error
    def self.mae(predictions, actuals)
      n = predictions.size
      sum = predictions.zip(actuals).map { |p, r| (p - r).abs }.sum
      sum / n.to_f
    end

    # Precision
    def self.precision(tp, fp)
      return 0.0 if (tp + fp).zero?
      tp.to_f / (tp + fp)
    end

    # Recall
    def self.recall(tp, fn)
      return 0.0 if (tp + fn).zero?
      tp.to_f / (tp + fn)
    end

    # F1 Score
    def self.f1(precision, recall)
      return 0.0 if (precision + recall).zero?
      2 * (precision * recall) / (precision + recall)
    end

    # Confusion Matrix
    def self.confusion_matrix(predictions, actuals, threshold = 0.5)
      tp = fp = tn = fn = 0
      predictions.zip(actuals).each do |pred, actual|
        pred_bin = pred > threshold ? 1 : 0
        case [pred_bin, actual]
        when [1, 1] then tp += 1
        when [1, 0] then fp += 1
        when [0, 0] then tn += 1
        when [0, 1] then fn += 1
        end
      end
      { tp: tp, fp: fp, tn: tn, fn: fn }
    end

    # Area Under the ROC Curve
    def self.auc_roc(predictions, actuals)
      pairs = predictions.zip(actuals).sort_by { |pred, _| -pred }
      positives = actuals.count(1)
      negatives = actuals.count(0)
      return 0.5 if positives == 0 || negatives == 0

      auc = 0.0
      fp = 0
      pairs.each do |_, actual|
        if actual == 1
          auc += fp
        else
          fp += 1
        end
      end
      auc / (positives * negatives).to_f
    end

    # Accuracy
    def self.accuracy(predictions, actuals, threshold = 0.5)
      correct = predictions.zip(actuals).count { |pred, actual| (pred > threshold ? 1 : 0) == actual }
      correct.to_f / predictions.size
    end
  end
end
