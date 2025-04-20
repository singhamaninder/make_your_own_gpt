import numpy as np

def create_dataset(sequence, window_size):
    x, y = [], []
    for i in range(len(sequence) - window_size):
        x.append(sequence[i:i+window_size])
        y.append(sequence[i+1:i+window_size+1])
    return np.array(x), np.array(y)
