import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.express as px
import pandas as pd
import numpy as np
from tokenizer import SimpleTokenizer
from models import TextAutoEncoder

st.set_page_config(page_title="Text Embedding Visualizer", layout="wide")

@st.cache_resource
def get_model_and_tokenizer():
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
    
    model = TextAutoEncoder(vocab_size, embedding_dim, hidden_dim)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Simple training loop to overfit
    data = torch.stack([tokenizer.encode(s, max_len) for s in sentences])
    model.train()
    for _ in range(200):
        optimizer.zero_grad()
        output = model(data, max_len)
        loss = criterion(output.view(-1, vocab_size), data.view(-1))
        loss.backward()
        optimizer.step()
        
    model.eval()
    return model, tokenizer, max_len

st.title("🚀 Text Embedding Journey")
st.markdown("""
This tool visualizes the **Journey of Text** as it is processed by a neural network into a numerical vector.
""")

model, tokenizer, max_len = get_model_and_tokenizer()

input_text = st.text_input("Enter a sentence to visualize its journey:", value="hello how are you")

if input_text:
    with torch.no_grad():
        input_tensor = tokenizer.encode(input_text, max_len).unsqueeze(0)
        journey = model.encoder.forward_with_journey(input_tensor)
        
        token_indices = input_tensor.squeeze(0).tolist()
        tokens = [tokenizer.idx2word.get(idx, "<UNK>") for idx in token_indices]
        
        # 1. Tokenization Step
        st.header("1. Tokenization")
        st.write("First, the text is split into tokens and mapped to vocabulary indices.")
        # Fix Arrow serialization by ensuring everything is string-friendly
        token_df = pd.DataFrame({"Token": tokens, "Index": [str(i) for i in token_indices]})
        transposed_df = token_df.T
        transposed_df.columns = [f"Pos {i}" for i in range(len(tokens))]
        st.table(transposed_df)

        # 2. Word Embeddings
        st.header("2. Word Embeddings")
        st.write("Each token index is converted into a high-dimensional vector (Embedding Dim = 16).")
        embeddings = journey["token_embeddings"].squeeze(0).numpy()
        
        fig_emb = px.imshow(embeddings, 
                            labels=dict(x="Embedding Dimension", y="Token Position", color="Value"),
                            x=[f"Dim {i}" for i in range(16)],
                            y=tokens,
                            color_continuous_scale='RdBu',
                            title="Initial Word Embeddings")
        st.plotly_chart(fig_emb, width='stretch')

        # 3. RNN Processing (The Journey)
        st.header("3. The RNN Journey (GRU)")
        st.write("""
        The RNN (GRU) processes the tokens one by one. 
        At each step, it combines the **current word** with its **previous memory** to update its hidden state.
        """)
        
        hidden_states = journey["all_hidden_states"].squeeze(0).numpy()
        
        step = st.slider("Step through the processing journey:", 0, max_len - 1, 0)
        
        col_st1, col_st2 = st.columns([1, 1])
        
        with col_st1:
            st.subheader(f"Step {step}: Processing '{tokens[step]}'")
            st.write(f"The model is now incorporating the word **'{tokens[step]}'** into its internal representation.")
            
            # Show the current hidden state
            current_hidden = hidden_states[step]
            reshaped_hidden = current_hidden.reshape(4, 8)
            fig_h = px.imshow(reshaped_hidden,
                              labels=dict(color="Value"),
                              color_continuous_scale='Viridis',
                              title=f"Hidden State at Step {step}")
            st.plotly_chart(fig_h, width='stretch')

        with col_st2:
            # Line chart showing how specific dimensions evolve
            st.subheader("Evolution of State")
            dim_to_track = st.multiselect("Select dimensions to track over time:", 
                                         options=list(range(32)), 
                                         default=[0, 10, 20, 30])
            
            evo_data = []
            for t in range(max_len):
                for d in dim_to_track:
                    evo_data.append({"Step": t, "Token": tokens[t], "Dimension": f"Dim {d}", "Value": hidden_states[t][d]})
            
            fig_evo = px.line(pd.DataFrame(evo_data), x="Step", y="Value", color="Dimension", 
                              hover_data=["Token"], title="Dimension Values Over Time")
            st.plotly_chart(fig_evo, width='stretch')

        # 4. Final Embedding
        st.header("4. Final Result: The Sentence Vector")
        st.write("The state after processing the last token (or <EOS>) is our final embedding.")
        
        final_embedding = journey["final_embedding"].numpy()
        st.code(final_embedding)
        
        # 5. Reconstruction (Verification)
        st.header("5. Verification: Decoding back to Text")
        # journey["final_embedding"] is already [1, 32], decoer expects [batch, hidden]
        reconstructed = model.decoder(journey["final_embedding"], max_len)
        decoded = tokenizer.decode(reconstructed.argmax(2).squeeze(0))
        st.success(f"**Decoded back to:** {decoded}")

st.sidebar.title("How it works")
st.sidebar.markdown("""
- **Token**: Individual word or symbol.
- **Word Embedding**: Fixed-size vector for a single word.
- **Hidden State**: The RNN's 'memory' that evolves as it reads each word.
- **Sentence Vector**: The final memory state that represents the whole sentence.
""")
