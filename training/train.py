import os
from training.tokenizer_util import train_and_save_tokenizer
from training.dataset import create_dataset
from models.gpt_model import build_gpt_model

import tensorflow as tf

def run_training():
    with open("data/data.txt", 'r', encoding='utf-8') as f:
        text_data = f.read()

    vocab_size = 5000
    max_seq_len = 100
    embed_dim = 256
    num_heads = 8
    ff_dim = 1024
    num_layers = 10
    batch_size = 100
    epochs = 4

    tokenizer = train_and_save_tokenizer(text_data, vocab_size, "saved/tokenizer.pkl")
    sequence = tokenizer.texts_to_sequences([text_data])[0]
    x_data, y_data = create_dataset(sequence, max_seq_len)

    model = build_gpt_model(vocab_size, max_seq_len, embed_dim, num_heads, ff_dim, num_layers)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_data, y_data, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    model.save("saved/gpt_model.keras")
    model.summary()
