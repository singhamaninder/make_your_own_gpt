
# 🤖 Make Your Own GPT

Welcome to **Make Your Own GPT** – a full-stack, production-ready GPT-style text generation system built using TensorFlow and Keras. Train your own mini GPT on custom text data and generate human-like text using just Python!

---

## 🚀 Features

- Tokenization with OOV handling  
- Custom sequence windowing  
- Transformer block with multi-head attention  
- Positional encoding for sequential context  
- End-to-end training pipeline  
- Softmax sampling with temperature control  
- Save/load model and tokenizer  
- Plug-and-play text generation script  

---

## 📁 Project Structure

```
Make-Your-Own-GPT/
├── data.txt                   # Training data
├── tokenizer.pkl              # Saved tokenizer
├── gpt_test_genai_class.keras  # Trained model
├── train.py                   # Training logic
├── generate.py                # Text generation script
├── model.py                   # Model architecture
├── utils.py                   # Data utilities
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git exclusions
└── README.md                  # Project documentation
```

---

## ⚙️ Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/make-your-own-gpt.git
cd make-your-own-gpt

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

1. Put your training data in `data.txt`
2. Run:

```bash
python train.py
```

This will:
- Tokenize text  
- Create training sequences  
- Build and train the transformer  
- Save the model and tokenizer  

---

## 📝 Generate Text

```bash
python generate.py
```

You can customize the prompt inside `generate.py`:

```python
seed_text = "Intermarche - AI Chatbot"
generated = generate_text(seed_text, model, tokenizer, num_tokens=500, temperature=1.0)
print(generated)
```

---

## 🔧 Model Settings

Edit these in `train.py` or `model.py`:

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

## 🧠 Sample Output

```
Input: Intermarche - AI Chatbot

Output:
Intermarche - AI Chatbot is an intelligent assistant built to support customers by leveraging generative models and real-time feedback. The system learns continuously and adapts to evolving queries...
```

---

## 📦 Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- tqdm (optional)

---

## 📜 License

MIT License. Free to use and modify.

---

## 👨‍💻 Author

Made by **Amaninder**  
AI/ML Developer | GPT Enthusiast | Builder of Smart Systems  
[LinkedIn](https://www.linkedin.com/in/amaninder-singh-826613112/) • speaktoamaninder@gmail.com
