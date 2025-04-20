import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

def train_and_save_tokenizer(text, vocab_size, path):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts([text])
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f)
    return tokenizer

def load_tokenizer(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
