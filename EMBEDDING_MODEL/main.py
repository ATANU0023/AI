import torch
import torch.nn as nn
import torch.optim as optim
from tokenizer import SimpleTokenizer
from models import TextAutoEncoder

def train_demonstration():
    # 1. Prepare Data
    sentences = [
        "hello how are you",
        "i am fine thank you",
        "this is a test",
        "building an embedding model",
        "convert text to vectors",
        "convert vectors to text"
    ]
    
    tokenizer = SimpleTokenizer(sentences)
    vocab_size = tokenizer.vocab_size
    embedding_dim = 16
    hidden_dim = 32
    max_len = 10
    
    # 2. Initialize Model
    model = TextAutoEncoder(vocab_size, embedding_dim, hidden_dim)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore <PAD>
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print(f"Vocab size: {vocab_size}")
    print("Training on sample data...")
    
    # Encode all sentences
    data = torch.stack([tokenizer.encode(s, max_len) for s in sentences])
    
    # 3. Simple Training Loop
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        output = model(data, max_len)
        
        # Reshape output and data for loss calculation
        # output: [batch, seq_len, vocab_size] -> [batch * seq_len, vocab_size]
        # data: [batch, seq_len] -> [batch * seq_len]
        loss = criterion(output.view(-1, vocab_size), data.view(-1))
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
            
    # 4. Demonstration
    model.eval()
    print("\n--- Demonstration ---")
    
    test_text = "hello how are you"
    print(f"Input Text: '{test_text}'")
    
    # Text to Vector (Encoder)
    with torch.no_grad():
        input_tensor = tokenizer.encode(test_text, max_len).unsqueeze(0)
        vector = model.encoder(input_tensor)
        print(f"Vector (Embedding) shape: {vector.shape}")
        print(f"Vector sample: {vector[0][:5]}...") # Show first 5 dimensions
        
        # Vector to Text (Decoder)
        reconstructed = model.decoder(vector, max_len)
        predicted_indices = reconstructed.argmax(2).squeeze(0)
        decoded_text = tokenizer.decode(predicted_indices)
        print(f"Decoded Text: '{decoded_text}'")

if __name__ == "__main__":
    train_demonstration()
