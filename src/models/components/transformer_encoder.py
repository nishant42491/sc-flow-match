import torch
import torch.nn as nn

import math


class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.embedder_1 = nn.Linear(input_dim, embed_dim)
        self.activation = nn.SELU()
        self.embedder_2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.activation(self.embedder_1(x))
        x = self.embedder_2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.decoder_1 = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.SELU()
        self.decoder_2 = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.activation(self.decoder_1(x))
        x = self.decoder_2(x)
        return x





class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TransformerEncoderLayer(nn.Module):
    def __init__(self, time_dim, input_dim, nhead, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        d_model = input_dim + time_dim
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        self.timestep_embedding = SinusoidalPosEmb(time_dim)
        self.final_linear = nn.Linear(d_model, input_dim)

    def forward(self, t, src, src_mask=None):

        #t is a 1 dimentional tensor of shape (batch_size) scale each elemnt in t by 100
        t = torch.round(t * 100)
        src = torch.cat([src,self.timestep_embedding(t)], axis = 1)
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        src = self.final_linear(src)

        return src


class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, embed_dim,
                 time_dim, num_layers, n_heads,
                 output_dim, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(input_dim, embed_dim)
        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(input_dim=embed_dim,
                                     time_dim=time_dim,
                                     nhead=n_heads,
                                     dropout=dropout) for _ in range(num_layers)]
        )
        self.decoder = Decoder(embed_dim, output_dim)

    def forward(self, t, x, args=None, kwargs=None, tr=False):

        if not tr:

            if t.dim() == 0 or t.dim() == 1:
                t = t.repeat(x.shape[0])


        encoded = self.encoder(x)

        for layer in self.transformer_layers:
            encoded = layer(t, encoded)

        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    input_dim = 2048
    output_dim =  2048
    embed_dim = 32
    time_dim = 32
    n_heads = 4
    num_layers = 4

    src = torch.rand(32, input_dim)
    t = torch.rand(32)
    model = TransformerAutoencoder(input_dim, embed_dim, time_dim,
                                   num_layers, n_heads,output_dim)

    output = model(t, src)
    print(output.shape)



