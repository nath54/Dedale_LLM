import torch
from torch import nn, Tensor
from transformers import AutoTokenizer
# import os
import math

PRINTDIMS = False

MAX_LOADED_BLOCKS = 2
BLOCKS_TO_REMOVE = 1

CONTEXT_LENGTH = 16
EMBEDDING_DIM = 16
HIDDEN_DIM = EMBEDDING_DIM
VOCAB_SIZE = 65024
NUMBER_OF_BLOCKS = 2
NUMBER_OF_ATTENTION_HEADS = 1

# device = torch.device("cpu")
# if torch.cuda.is_available():
#     torch.device("cuda")
device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b-instruct")
tokenizer.pad_token = tokenizer.eos_token
vocab = tokenizer.get_vocab()


def printd(*args):
    if PRINTDIMS:
        print("\u001b[34m * ", *args, "\u001b[m")


class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)

    def forward(self, X):
        printd("Lib->Embedding, line 39 : ", X.shape, type(X), X.type())
        X = self.embedding(X)
        printd("Lib->Embedding, line 41 : ", X.shape, type(X), X.type())
        return X


class Routeur(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(CONTEXT_LENGTH * EMBEDDING_DIM,
                             NUMBER_OF_BLOCKS + 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, X):
        printd("Lib->Routeur, line 53, X : ", X.shape, X.type())
        X = torch.flatten(X, start_dim=0)
        printd("Lib->Routeur, line 55, X : ", X.shape, X.type())
        X = self.lin(X)
        printd("Lib->Routeur, line 57, X : ", X.shape, X.type())
        X = self.softmax(X)
        printd("Lib->Routeur, line 59, X : ", X.shape, X.type())
        idx = torch.multinomial(X, 1)
        printd("Lib->Routeur, line 61, ids : ",
               idx, type(idx), idx.type())
        return idx


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(EMBEDDING_DIM, HIDDEN_DIM)
        self.gelu = nn.GELU()
        self.lin2 = nn.Linear(HIDDEN_DIM, EMBEDDING_DIM)

    def forward(self, X):
        printd("Lib->FeedForward, Line 74, X : ", X.shape, X.type())
        X = self.lin1(X)
        printd("Lib->FeedForward, Line 76, X : ", X.shape, X.type())
        X = self.gelu(X)
        printd("Lib->FeedForward, Line 78, X : ", X.shape, X.type())
        X = self.lin2(X)
        printd("Lib->FeedForward, Line 80, X : ", X.shape, X.type())
        return X


class NextTokenPrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(EMBEDDING_DIM*CONTEXT_LENGTH, VOCAB_SIZE)
        self.softmax = nn.Softmax(dim=0)

    def get_next_token_index(self, X, k, top_k_value):
        """
        Args:
            X: A tensor of shape (vocabulary_size, )
            k: The number of tokens to consider for the next token
            top_k_value: The minimum value for a token to be considered

        Returns:
            The index of the next token
        """

        printd("Lib->NextTokenPrediction, Line 101, X : ", X.shape, X.type())
        topk, indices = torch.topk(X, k)
        filtered_indices = indices
        if len(filtered_indices) == 0:
            return torch.multinomial(X, 1)
        else:
            return torch.multinomial(X[filtered_indices], 1)

    def forward(self, X, topk_sorted_index_limit=None):
        X = torch.flatten(X, start_dim=0)
        printd("Lib->NextTokenPrediction, Line 111, X : ", X.shape, X.type())
        X = self.lin(X)
        printd("Lib->NextTokenPrediction, Line 113, X : ", X.shape, X.type())
        X = self.softmax(X)
        printd("Lib->NextTokenPrediction, Line 115, X : ", X.shape, X.type())

        return X

    def get_next_token_idx(self, X: Tensor, topk_sorted_index_limit=None):
        if topk_sorted_index_limit is int:
            idx = self.get_next_token_index(X, topk_sorted_index_limit)
        else:
            idx = torch.multinomial(X, 1)

        return idx


class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.query_linear = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        # * NUMBER_OF_ATTENTION_HEADS)
        self.key_linear = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        # * NUMBER_OF_ATTENTION_HEADS)
        self.value_linear = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        # * NUMBER_OF_ATTENTION_HEADS)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X):
        """
        Args:
            X: A tensor of shape (BATCH_SIZE, CONTEXT_LENGTH, EMBEDDING_DIM)

        Returns:
            A tensor of shape (BATCH_SIZE, CONTEXT_LENGTH, EMBEDDING_DIM)
        """

        printd("Lib->MultiHeadSelfAttention, Line 147, X : ",
               X.shape, X.type())

        if (len(X.shape) == 2):
            _, _ = X.shape

            query = self.query_linear(X)
            printd("Lib->MultiHeadSelfAttention, Line 152, Query : ",
                   query.shape)
            query = query.view(CONTEXT_LENGTH,
                               NUMBER_OF_ATTENTION_HEADS,
                               HIDDEN_DIM // NUMBER_OF_ATTENTION_HEADS)
            printd("Lib->MultiHeadSelfAttention, Line 156, Query : ",
                   query.shape)
            key = self.key_linear(X)
            printd("Lib->MultiHeadSelfAttention, Line 158, Key : ", key.shape)
            key = key.view(CONTEXT_LENGTH, NUMBER_OF_ATTENTION_HEADS,
                           HIDDEN_DIM // NUMBER_OF_ATTENTION_HEADS)
            printd("Lib->MultiHeadSelfAttention, Line 161, Key : ", key.shape)
            value = self.value_linear(X)
            printd("Lib->MultiHeadSelfAttention, Line 163, Value : ",
                   value.shape)
            value = value.view(CONTEXT_LENGTH,
                               NUMBER_OF_ATTENTION_HEADS,
                               HIDDEN_DIM // NUMBER_OF_ATTENTION_HEADS)
            printd("Lib->MultiHeadSelfAttention, Line 167, Value : ",
                   value.shape)

            attention = torch.matmul(query,
                                     key.transpose(-1, -2)) / math.sqrt(
                                     HIDDEN_DIM // NUMBER_OF_ATTENTION_HEADS)
            attention = self.softmax(attention)

            output = torch.matmul(attention, value).view(CONTEXT_LENGTH,
                                                         HIDDEN_DIM)
            printd("Lib->MultiHeadSelfAttention, Line 177, out : ",
                   output.shape)
        else:
            batch_size, _, _ = X.shape

            query = self.query_linear(X)
            printd("Lib->MultiHeadSelfAttention, Line 152, Query : ",
                   query.shape)
            query = query.view(batch_size, CONTEXT_LENGTH,
                               NUMBER_OF_ATTENTION_HEADS,
                               HIDDEN_DIM // NUMBER_OF_ATTENTION_HEADS)
            printd("Lib->MultiHeadSelfAttention, Line 156, Query : ",
                   query.shape)
            key = self.key_linear(X)
            printd("Lib->MultiHeadSelfAttention, Line 158, Key : ", key.shape)
            key = key.view(batch_size, CONTEXT_LENGTH,
                           NUMBER_OF_ATTENTION_HEADS,
                           HIDDEN_DIM // NUMBER_OF_ATTENTION_HEADS)
            printd("Lib->MultiHeadSelfAttention, Line 161, Key : ", key.shape)
            value = self.value_linear(X)
            printd("Lib->MultiHeadSelfAttention, Line 163, Value : ",
                   value.shape)
            value = value.view(batch_size, CONTEXT_LENGTH,
                               NUMBER_OF_ATTENTION_HEADS,
                               HIDDEN_DIM // NUMBER_OF_ATTENTION_HEADS)
            printd("Lib->MultiHeadSelfAttention, Line 167, Value : ",
                   value.shape)

            attention = torch.matmul(query,
                                     key.transpose(-1, -2)) / math.sqrt(
                                     HIDDEN_DIM // NUMBER_OF_ATTENTION_HEADS)
            attention = self.softmax(attention)

            output = torch.matmul(attention, value).view(batch_size,
                                                         CONTEXT_LENGTH,
                                                         HIDDEN_DIM)
            printd("Lib->MultiHeadSelfAttention, Line 177, out : ",
                   output.shape)

        return output


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = MultiHeadSelfAttention()
        self.ln1 = nn.LayerNorm((EMBEDDING_DIM))
        self.ff = FeedForward()
        self.ln2 = nn.LayerNorm((EMBEDDING_DIM))

    def forward(self, X):
        printd("Lib->Block, Line 191, X : ", X.shape, X.type())
        y = self.attention(X)
        printd("Lib->Block, Line 193, X : ", X.shape, X.type())
        X = self.ln1(X+y)
        printd("Lib->Block, Line 195, X : ", X.shape, X.type())
        y = self.ff(X)
        printd("Lib->Block, Line 197, X : ", X.shape, X.type())
        X = self.ln2(X+y)
        printd("Lib->Block, Line 199, X : ", X.shape, X.type())
        return X
