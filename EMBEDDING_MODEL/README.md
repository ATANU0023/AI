# Text Embedding Autoencoder

This project implements a custom Text-to-Vector (Embedding) model and a Vector-to-Text (Decoder) model using PyTorch. It demonstrates how text sequences can be compressed into fixed-size numerical representations and subsequently reconstructed.

## Project Structure

- `tokenizer.py`: A simple word-level tokenizer that maps unique words to integer indices. It handles special tokens like `<PAD>` (padding), `<SOS>` (start of sentence), `<EOS>` (end of sentence), and `<UNK>` (unknown words).
- `models.py`: Contains the neural network architectures:
    - **TextEncoder**: A GRU-based model that processes word embeddings and outputs the final hidden state as the "sentence embedding".
    - **TextDecoder**: A GRU-based model that takes a sentence embedding and iteratively generates the most likely next word until the sequence is completed.
    - **TextAutoEncoder**: A wrapper that connects the Encoder and Decoder for end-to-end training.
- `main.py`: A demonstration script that creates a small vocabulary, trains the autoencoder on sample sentences, and demonstrates the conversion process.
- `app.py`: An interactive Streamlit dashboard for real-time text-to-embedding visualization, including a **Step-by-Step Journey** visualization.
- `requirements.txt`: List of Python dependencies (`torch`, `numpy`, `streamlit`, `plotly`).

## How It's Made

### 1. The Tokenizer
The tokenizer builds a vocabulary from the training sentences. It converts text into a sequence of integers and adds padding to ensure uniform input length for the neural network.

### 2. The Encoder (Text -> Vector)
- **Embedding Layer**: Converts word indices into dense vectors.
- **GRU (Gated Recurrent Unit)**: Processes the sequence of word vectors. The final hidden state of the GRU captures the semantic context of the entire sentence, resulting in a fixed-size vector (e.g., 32 dimensions).

### 3. The Decoder (Vector -> Text)
- The decoder takes the fixed-size vector from the encoder as its initial hidden state.
- It starts with a `<SOS>` token and predicts the next word in the sequence.
- The predicted word is fed back into the next step until an `<EOS>` token is generated or the maximum length is reached.

## How It Works

1. **Encoding**:
   ```python
   embedding = encoder(tokenized_text) # Output: [1, 32] dimensional vector
   ```
2. **Decoding**:
   ```python
   reconstructed_text = decoder(embedding) # Output: Original sentence
   ```

## Getting Started

### Prerequisites
- Python 3.7+
- PyTorch
- NumPy

### Installation
```bash
pip install -r requirements.txt
```

### Running the Demo (CLI)
```bash
python main.py
```

### Running the Interactive Dashboard (Streamlit)
```bash
python -m streamlit run app.py
```

This will open a web interface where you can enter text and see the corresponding embedding heatmap and reconstructed text in real-time.

