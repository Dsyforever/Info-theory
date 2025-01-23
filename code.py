import re
import numpy as np
from collections import Counter, defaultdict


def preprocess_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()[244:]  # Skip the first 244 lines
        text = ''.join(lines)

    # Convert all letters to uppercase
    text = text.upper()

    # Replace all non-letter characters with "$"
    text = ''.join(char if char.isalpha() else '$' for char in text)

    # Replace consecutive "$" with a single "$"
    text = re.sub(r'\$+', '$', text)

    return text


def calculate_frequencies(text):
    px = Counter(text)
    xy = [text[i:i + 2] for i in range(len(text) - 1)]
    pxy = Counter(xy)

    # Calculate Py from Pxy
    py = defaultdict(int)
    for pair in pxy:
        py[pair[1]] += pxy[pair]

    return px, py, pxy


def calculate_entropy(freq_dict, total_count):
    entropy = 0
    for count in freq_dict.values():
        prob = count / total_count
        entropy -= prob * np.log2(prob + 1e-10)  # Add a small value to avoid log(0)
    return entropy


def conditional_entropy(pxy, px, total_pairs):
    h_y_given_x = 0
    for pair, count in pxy.items():
        prob_xy = count / total_pairs
        prob_x = px[pair[0]] / (total_pairs + 1)
        h_y_given_x -= prob_xy * np.log2(prob_xy / (prob_x + 1e-10) + 1e-10)
    return h_y_given_x


def mutual_information(h_x, h_y, h_y_given_x):
    return h_y - h_y_given_x


def main():
    filename = 'shakespeare.txt'

    # Preprocess the file
    processed_text = preprocess_file(filename)

    # Calculate frequencies
    px, py, pxy = calculate_frequencies(processed_text)

    # Total counts
    total_chars = sum(px.values())
    total_pairs = sum(pxy.values())

    # Calculate entropies
    H_X = calculate_entropy(px, total_chars)
    H_Y = calculate_entropy(py, total_chars)
    H_Y_given_X = conditional_entropy(pxy, px, total_chars)
    I_XY = mutual_information(H_X, H_Y, H_Y_given_X)

    print(f"H(X): {H_X}")
    print(f"H(Y): {H_Y}")
    print(f"H(Y|X): {H_Y_given_X}")
    print(f"I(X;Y): {I_XY}")


if __name__ == '__main__':
    main()
