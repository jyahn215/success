import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super(SimpleModel, self).__init__()
        
        # Embedding layer with frozen weights
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Freeze the embedding layer
        for param in self.embedding.parameters():
            param.requires_grad = False
        
        # Linear layer to process the embedding
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        # Get the embeddings for the input
        embedded = self.embedding(x)
        
        # Pass the embedding through the linear layer
        output = self.fc(embedded)
        
        return output

# Example usage
vocab_size = 1000  # Vocabulary size
embedding_dim = 300  # Embedding dimension
output_dim = 10  # Output dimension for linear layer

# Instantiate the model
model = SimpleModel(vocab_size, embedding_dim, output_dim)

# Example input tensor (batch size 1, one token)
input_tensor = torch.LongTensor([i for i in range(vocab_size)])  # Input index for a single token
output = model(input_tensor)

print(output)