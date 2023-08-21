import torch 
from torch import nn 

# You can, by hand, make a skip-gram network to encode the random walks 
class SkipGram(nn.Module):
    def __init__(self, num_nodes, enc_dim):
        super().__init__()
        self.num_nodes = num_nodes

        # Torch actually has a specific implementation for this 
        # called nn.LookupTable, but for the sake of doing it by 
        # hand, just use a linear layer
        self.enc = nn.Linear(num_nodes, enc_dim)
        self.dec = nn.Sequential(
            nn.Linear(enc_dim, num_nodes), 
            nn.Softmax(dim=1)
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, target, rw): 
        walk_len = rw.size(1)

        # Convert from e.g. [0,1,2] to [0,0,0,1,1,1,2,2,2]
        target = target.repeat_interleave(walk_len)

        # Get encoding of elements in random walk (|batch| x enc_dim)
        input_x = torch.eye(self.num_nodes)[rw.flatten()]
        walk_emb = self.enc(input_x)
        
        # Project it to node predictions (|batch| x |N|)
        decoded = self.dec(walk_emb)
        
        # Train s.t. embedding for nodes in same walk is 
        # similar to target node
        loss = self.loss(decoded, target)
        return loss 

    def embed(self, batch):
        input_x = torch.eye(self.num_nodes)[batch]
        return self.enc(input_x)


if __name__ == '__main__':
    rws = torch.tensor([
        [0,1,2,3],
        [2,3,4,5],
        [1,2,3,4]
    ])
    batch = torch.tensor([0,1,2])
    sg = SkipGram(6,2)
    l = sg.forward(batch, rws)
    print(l)