import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# reference
# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, sentence_len=128):
        super(PositionalEncoding, self).__init__()
        self.positionalEncoding = torch.zeros((sentence_len, embedding_dim))

        for pos in range(0, sentence_len):
            for i in range(0, embedding_dim // 2):
                self.positionalEncoding[pos, 2 * i] = math.sin(pos / math.pow(10000, 2 * i / embedding_dim))
                self.positionalEncoding[pos, 2 * i + 1] = math.cos(pos / math.pow(10000, 2 * i / embedding_dim))

        self.register_buffer('positional_encoding', self.positionalEncoding)

    def forward(self, x):
        sentence_len = x.size(1)
        out = x + self.positionalEncoding[:sentence_len, :].to(x)
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dk):
        super(ScaledDotProductAttention, self).__init__()
        self.dk = dk
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask):
        out = torch.matmul(query, key.transpose(2, 3))
        out /= math.sqrt(self.dk)

        if mask is not None:
            out = out.masked_fill(mask == 0, -1e10)
        out = self.softmax(out)

        out = torch.matmul(out, value)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.head = head
        self.dk = embedding_dim // head

        self.dense_query = nn.Linear(embedding_dim, embedding_dim)
        self.dense_key = nn.Linear(embedding_dim, embedding_dim)
        self.dense_value = nn.Linear(embedding_dim, embedding_dim)

        self.scaled_dot_product_attention = ScaledDotProductAttention(dk=self.dk)
        self.dropout = nn.Dropout(dropout_rate)

        self.dense = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, query, key, value, mask=None):
        batch, _, embedding_dim = query.size()

        query_out = self.dense_query(query).view(batch, -1, self.head, self.dk).transpose(1, 2)
        key_out = self.dense_key(key).view(batch, -1, self.head, self.dk).transpose(1, 2)
        value_out = self.dense_value(value).view(batch, -1, self.head, self.dk).transpose(1, 2)

        out = self.scaled_dot_product_attention(query_out, key_out, value_out, mask).transpose(1, 2).contiguous().view(
            batch, -1, self.embedding_dim)

        out = self.dropout(out)
        out = self.dense(out)

        return out


class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.embedding_dim = embedding_dim
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(self.embedding_dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(self.embedding_dim), requires_grad=True)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        norm = self.gamma * (x - mean) / (std + self.eps) + self.beta

        return norm


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.embedding_dim = embedding_dim
        self.inner_dim = 2048

        # in the paper, they suggest two kind of methods.
        self.dense = nn.Sequential(
            nn.Linear(self.embedding_dim, self.inner_dim),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(self.inner_dim, self.embedding_dim),
            nn.Dropout(dropout_rate)
        )

        # If you use Conv layer, you have to adjust the dimension.
        # self.conv = nn.Sequential(
        #     nn.Conv2d(self.embedding_dim, self.inner_dim, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Conv2d(self.inner_dim, self.embedding_dim, kernel_size=1)
        # )

    def forward(self, x):
        out = self.dense(x)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, head, embedding_dim, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.head = head
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate

        self.multi_head_attention = MultiHeadAttention(head, embedding_dim, dropout_rate)

        self.layer_norm1 = LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, dropout_rate)
        self.layer_norm2 = LayerNorm(embedding_dim)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, x_mask):
        # multi head attention
        multi_head_out = self.dropout1(self.multi_head_attention(x, x, x, x_mask))
        multi_head_out = self.layer_norm1(multi_head_out + x)

        # feed forward layer
        feed_forward_out = self.dropout2(self.feed_forward(multi_head_out))
        feed_forward_out = self.layer_norm2(feed_forward_out + multi_head_out)
        return feed_forward_out


class Encoder(nn.Module):
    def __init__(self, input_size, max_len, heads, embedding_dim, N, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.max_len = max_len
        self.N = N
        encoderLayers = []

        # Embedding shape, (max_num of inputs(voca_size), embedding_dim)
        self.token_embedding = nn.Embedding(input_size, embedding_dim)

        # segment embedding
        self.segment_embedding = nn.Embedding(3, embedding_dim)

        # PositionalEncoding
        self.positional_encoding = PositionalEncoding(embedding_dim)

        for _ in range(N):
            encoderLayers.append(EncoderLayer(heads, embedding_dim, dropout_rate))
        self.encoder = nn.Sequential(*encoderLayers)

    def forward(self, x, mask, segment):
        token_embedding = self.token_embedding(x)
        segment_embedding = self.segment_embedding(segment)
        out = token_embedding + self.positional_encoding(token_embedding) + segment_embedding

        # N time iteration
        for _ in range(self.N):
            out = self.encoder[_](out, mask)
        return out


class BERT(nn.Module):
    def __init__(self, input_vocab, max_len, heads, embedding_dim, N, dropout_rate=0.1):
        super(BERT, self).__init__()
        self.encoder = Encoder(input_vocab, max_len, heads, embedding_dim, N, dropout_rate)

        self.mlm_out = nn.Sequential(
            nn.Linear(embedding_dim, input_vocab),
        )

        self.nsp_out = nn.Sequential(
            nn.Linear(embedding_dim, 2)
        )

    def forward(self, x, segment):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        out = self.encoder(x, mask, segment)

        return self.mlm_out(out), self.nsp_out(out[:, 0])
