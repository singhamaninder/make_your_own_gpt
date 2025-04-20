import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from training.tokenizer_util import load_tokenizer
from utils.positional_encoding import PositionalEncoding

def generate_text(seed_text, model, tokenizer, max_seq_len, num_tokens=50, temperature=1.0):
    for _ in range(num_tokens):
        token_seq = tokenizer.texts_to_sequences([seed_text])[0]
        token_seq = token_seq[-max_seq_len:]
        padded_seq = pad_sequences([token_seq], maxlen=max_seq_len)

        preds = model.predict(padded_seq, verbose=0)[0, -1]
        preds = np.log(np.asarray(preds).astype('float64') + 1e-9) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))
        next_token = np.random.choice(len(preds), p=preds)
        next_word = tokenizer.index_word.get(next_token, '')

        seed_text += ' ' + next_word
        if not next_word:
            break
    return seed_text

def load_model_and_generate(seed_text):
    model = tf.keras.models.load_model("saved/gpt_model.keras", custom_objects={"PositionalEncoding": PositionalEncoding})
    tokenizer = load_tokenizer("saved/tokenizer.pkl")
    return generate_text(seed_text, model, tokenizer, max_seq_len=100)
