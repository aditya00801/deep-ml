from collections import Counter 
def unigram_probability(corpus: str, word: str) -> float:
    tokens= corpus.split()

    frequency = Counter(tokens)

    total_words = len(tokens)

    probability = frequency[word] / len(tokens)
    return probability
