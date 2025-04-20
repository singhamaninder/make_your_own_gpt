import tensorflow as tf
from tensorflow.keras import layers
from utils.positional_encoding import PositionalEncoding

def transformer_block(embed_dim, num_heads, ff_dim, dropout=0.1):
    inputs = layers.Input(shape=(None, embed_dim))
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + layers.Dropout(dropout)(attn_output))
    
    ffn_output = tf.keras.Sequential([
        layers.Dense(ff_dim, activation='relu'),
        layers.Dense(embed_dim)
    ])(out1)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + layers.Dropout(dropout)(ffn_output))
    
    return tf.keras.Model(inputs=inputs, outputs=out2)

def build_gpt_model(vocab_size, max_seq_len, embed_dim, num_heads, ff_dim, num_layers):
    inputs = layers.Input(shape=(max_seq_len,))
    x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)
    x = PositionalEncoding(max_seq_len, embed_dim)(x)
    for _ in range(num_layers):
        x = transformer_block(embed_dim, num_heads, ff_dim)(x)
    outputs = layers.Dense(vocab_size, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)
