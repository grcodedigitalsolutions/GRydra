module GRYDRA
  module Preprocessing
    module Text
      # Create vocabulary from texts
      def self.create_vocabulary(texts)
        texts.map(&:split).flatten.map(&:downcase).uniq
      end

      # Create advanced vocabulary with frequency filtering
      def self.create_advanced_vocabulary(texts, min_frequency = 1, max_words = nil)
        frequencies = Hash.new(0)
        texts.each do |text|
          text.downcase.split.each { |word| frequencies[word] += 1 }
        end

        vocabulary = frequencies.select { |_, freq| freq >= min_frequency }.keys

        if max_words && vocabulary.size > max_words
          vocabulary = frequencies.sort_by { |_, freq| -freq }.first(max_words).map(&:first)
        end

        vocabulary.sort
      end

      # Vectorize text (binary)
      def self.vectorize_text(text, vocabulary)
        vector = Array.new(vocabulary.size, 0)
        words = text.downcase.split
        words.each do |word|
          index = vocabulary.index(word)
          vector[index] = 1 if index
        end
        vector
      end

      # Vectorize text using TF-IDF
      def self.vectorize_text_tfidf(text, vocabulary, corpus_frequencies)
        vector = Array.new(vocabulary.size, 0.0)
        words = text.downcase.split
        doc_frequencies = Hash.new(0)

        words.each { |word| doc_frequencies[word] += 1 }

        vocabulary.each_with_index do |word, idx|
          next unless doc_frequencies[word] > 0

          tf = doc_frequencies[word].to_f / words.size
          idf = Math.log(corpus_frequencies.size.to_f / (corpus_frequencies[word] || 1))
          vector[idx] = tf * idf
        end

        vector
      end

      # Normalize with vocabulary
      def self.normalize_with_vocabulary(data, vocabulary)
        max_value = vocabulary.size
        data.map { |vector| vector.map { |v| v.to_f / max_value } }
      end
    end
  end
end
