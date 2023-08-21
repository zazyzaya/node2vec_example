import torch 
from torch import nn 

# You can, by hand, make a skip-gram network to encode the random walks 
class SkipGram(nn.Module):
    def __init__(self, num_nodes, enc_dim):
        self.super().__init__()

        # Torch actually has a specific implementation for this 
        # called nn.LookupTable, but for the sake of doing it by 
        # hand, just use a trainable param (it's essentially the same)
        self.encode = nn.Parameter(torch.random(num_nodes, enc_dim))
        self.decode = nn.Sequential(
            nn.Linear(enc_dim, num_nodes), 
            nn.Softmax(dim=1)
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, target, rw): 
        walk_len = rw.size(1)

        # Convert from e.g. [0,1,2] to [0,0,0,1,1,1,2,2,2]
        target = target.repeat_interleave(walk_len)

        # Get encoding of elements in random walk (|batch| x enc_dim)
        walk_emb = self.encode[rw.flatten()]
        
        # Project it to node predictions (|batch| x |N|)
        decoded = self.decode(walk_emb)
        
        # Train s.t. embedding for nodes in same walk is 
        # similar to target node
        loss = self.loss(decoded, target)
        return loss 

    def embed(self, batch):
        return self.encode[batch]
