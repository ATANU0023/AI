import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        
    def forward(self, x):
        embedded = self.embedding(x)
        # We only need the final hidden state as the "sentence embedding"
        _, hidden = self.gru(embedded)
        return hidden.squeeze(0)

    def forward_with_journey(self, x):
        # x shape: [batch, seq_len]
        embedded = self.embedding(x) # [batch, seq_len, embedding_dim]
        # output shape: [batch, seq_len, hidden_dim] - all hidden states
        # hidden shape: [1, batch, hidden_dim] - final hidden state
        output, hidden = self.gru(embedded)
        return {
            "token_embeddings": embedded,
            "all_hidden_states": output,
            "final_embedding": hidden.squeeze(0)
        }

class TextDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TextDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, encoder_hidden, target_len):
        batch_size = encoder_hidden.size(0)
        # Start with <SOS> token
        input_token = torch.tensor([[1]] * batch_size) # 1 is <SOS>
        
        hidden = encoder_hidden.unsqueeze(0)
        outputs = []
        
        for _ in range(target_len):
            embedded = self.embedding(input_token)
            output, hidden = self.gru(embedded, hidden)
            prediction = self.out(output.squeeze(1))
            outputs.append(prediction)
            
            # Use teacher forcing or top prediction (here simplified)
            input_token = prediction.argmax(1).unsqueeze(1).detach()
            
        return torch.stack(outputs, dim=1)

class TextAutoEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TextAutoEncoder, self).__init__()
        self.encoder = TextEncoder(vocab_size, embedding_dim, hidden_dim)
        self.decoder = TextDecoder(vocab_size, embedding_dim, hidden_dim)
        
    def forward(self, x, target_len):
        embedding = self.encoder(x)
        output = self.decoder(embedding, target_len)
        return output
