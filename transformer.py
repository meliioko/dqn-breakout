import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(TransformerBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.multihead_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=14, emb_size=256):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
            nn.Linear((84 // patch_size) * (84 // patch_size), emb_size)
        )

    def forward(self, x):
        x = self.projection(x).transpose(1, 2)  # B, T, N, E
        return x

class TransformerDQN(nn.Module):
    def __init__(self, action_size, d_model=256, nhead=4, num_layers=3, dim_feedforward=1024, dropout=0.1):
        super(TransformerDQN, self).__init__()

        # Patch embedding layer
        self.patch_embedding = PatchEmbedding(emb_size=d_model)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])

        # Output layer
        self.out = nn.Linear(d_model, action_size)

    def forward(self, x):
        x = self.patch_embedding(x)
        for transformer in self.transformer_blocks:
            x = transformer(x)
        x = x.mean(dim=1)  # Pooling
        return self.out(x)
