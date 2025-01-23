import re
import heapq
from collections import Counter
import numpy as np

def preprocess_file(filename, skip=0):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()[skip:]  # Skip initial lines if specified
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
    return px

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


def mutual_information(h_y, h_y_given_x):
    return h_y - h_y_given_x


def build_huffman_tree(px):
    # Create a priority queue where each element is (frequency, character)
    heap = [(freq, char) for char, freq in px.items()]
    heapq.heapify(heap)

    # Iterate until the heap has only one element
    while len(heap) > 1:
        # Pop the two smallest elements
        freq1, left = heapq.heappop(heap)
        freq2, right = heapq.heappop(heap)

        # Merge these two nodes and push back into the heap
        merged = (freq1 + freq2, (left, right))
        heapq.heappush(heap, merged)

    # The remaining element is the root of the Huffman tree
    return heap[0][1]


def calculate_average_length(huffman_tree, px):
    def get_lengths(tree, current_length):
        if isinstance(tree, str):
            # Leaf node, return its length
            return {tree: current_length}
        left, right = tree
        lengths = {}
        lengths.update(get_lengths(left, current_length + 1))
        lengths.update(get_lengths(right, current_length + 1))
        return lengths

    lengths = get_lengths(huffman_tree, 0)

    # Calculate the average length
    total_weighted_length = sum(px[char] * length for char, length in lengths.items())
    total_chars = sum(px.values())
    return total_weighted_length / total_chars


def main():
    filename = 'shakespeare.txt'
    # Preprocess the first file
    processed_text = preprocess_file(filename, 244)
    # Calculate frequencies
    px = calculate_frequencies(processed_text)
    # Build Huffman Tree
    huffman_tree = build_huffman_tree(px)
    # Calculate Huffman Code lengths
    avg_length_shakespeare = calculate_average_length(huffman_tree, px)
    entropy_s = calculate_entropy(px,sum(px.values()))
    print("Average Length of Shakespeare:", avg_length_shakespeare)
    print("Entropy of Shakespeare", entropy_s)
    # Preprocess the second file
    filename_h = 'holmes.txt'
    processed_text_h = preprocess_file(filename_h)
    # Calculate frequencies of the second text
    px_holmes = calculate_frequencies(processed_text_h)
    entropy_h=calculate_entropy(px_holmes,sum(px_holmes.values()))
    # Calculate average length for Holmes using lengths obtained from Shakespeare
    avg_length_holmes = calculate_average_length(huffman_tree, px_holmes)
    print("Average Length of Holmes (using Shakespeare codes):", avg_length_holmes)
    print("Entropy of Holmes is", entropy_h)

if __name__ == '__main__':
    main()