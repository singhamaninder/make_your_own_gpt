# Make Your Own GPT

A full-stack, production-ready GPT-style text generation system built with TensorFlow and Keras. Train your own mini GPT model on custom text data and generate human-like text using Python.

---

## Features

- Tokenization with out-of-vocabulary (OOV) handling  
- Custom sequence windowing for training data  
- Transformer architecture with multi-head attention  
- Positional encoding for capturing sequential context  
- End-to-end training pipeline with model saving/loading  
- Softmax sampling with temperature control for text generation  
- Modular and extensible codebase for easy customization  

---

## Project Structure

```
Make-Your-Own-GPT/
├── data/                      # Folder containing training data files
│   ├── data.txt               # Default training data
│   └── data_large.txt         # Larger training dataset
├── inference/                 # Text generation scripts
│   └── generate_text.py       # Script to generate text from trained model
├── models/                    # Model architecture files
│   └── gpt_model.py           # GPT model definition
├── training/                  # Training utilities and scripts
│   ├── dataset.py             # Dataset preparation utilities
│   ├── tokenizer_util.py      # Tokenizer utilities
│   └── train.py               # Training script
├── utils/                     # Helper utilities
│   └── positional_encoding.py # Positional encoding implementation
├── saved/                     # Folder for saved models and tokenizers
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
└── main.py                    # Main entry point (if applicable)
```

---

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/make-your-own-gpt.git
cd make-your-own-gpt
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training the Model

1. Place your training data in the `data/` folder (e.g., `data/data.txt`).

2. Run the training script:

```bash
python training/train.py
```

This will:
- Tokenize the text data  
- Create training sequences  
- Build and train the transformer model  
- Save the trained model and tokenizer in the `saved/` folder  

---

## Generating Text

Run the text generation script:

```bash
python inference/generate_text.py
```

You can customize the prompt inside `generate_text.py`:

```python
seed_text = "Your prompt here"
generated = generate_text(seed_text, model, tokenizer, num_tokens=500, temperature=1.0)
print(generated)
```

---

## Model Configuration

Adjust these parameters in `training/train.py` or `models/gpt_model.py` to customize the model:

```python
vocab_size = 5000
max_seq_len = 100
embed_dim = 256
num_heads = 8
ff_dim = 1024
num_layers = 10
batch_size = 100
epochs = 10
```

---

## Sample Output

```
Input: Your prompt here

Output:
Your prompt here is an intelligent assistant built to support users by leveraging generative models and real-time feedback. The system learns continuously and adapts to evolving queries...
```

---

## Requirements

- Python 3.8 or higher  
- TensorFlow 2.x  
- NumPy  
- tqdm (optional)  

---

## License

This project is licensed under the MIT License. Feel free to use and modify it.

---

## Author

**Amaninder**  
AI/ML Developer | GPT Enthusiast | Builder of Smart Systems  
[LinkedIn](https://www.linkedin.com/in/amaninder-singh-826613112/)  
Email: speaktoamaninder@gmail.com
