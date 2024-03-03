import torch
from torch import nn, Tensor
from transformers import AutoTokenizer
from custom_tokenisers import SingleCharactersTokenizer
import math
from inspect import currentframe
from config import PRINTDIMS, CONFIG_TOKENIZER


MAX_LOADED_BLOCKS = 5
BLOCKS_TO_REMOVE = 1

CONTEXT_LENGTH = 32
EMBEDDING_DIM = CONTEXT_LENGTH * 2
HIDDEN_DIM = EMBEDDING_DIM
NUMBER_OF_BLOCKS = 5
NUMBER_OF_ATTENTION_HEADS = 1

if CONFIG_TOKENIZER == "falcon_tokenizer":
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b-instruct")
elif CONFIG_TOKENIZER == "single_characters_tokenizer":
    tokenizer = SingleCharactersTokenizer()
else:
    print("\u001b[31mError: bad value of `CONFIG_TOKENIZER`.\u001b[m")
    exit()

tokenizer.pad_token = tokenizer.eos_token
vocab = tokenizer.get_vocab()
VOCAB_SIZE = len(vocab)
# print("Vocab size : ", VOCAB_SIZE)

# device = torch.device("cpu")
# if torch.cuda.is_available():
#     torch.device("cuda")
device = torch.device("cpu")


def lineno() -> int:
    cf = currentframe()
    return cf.f_back.f_lineno


def printd(*args):
    if PRINTDIMS:
        print("\u001b[34m * ", *args, "\u001b[m")


class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)

    def forward(self, X):
        printd(f"Lib->Embedding, l{lineno()}, X: ", X.shape, type(X), X.type())
        X = self.embedding(X)
        printd(f"Lib->Embedding, l{lineno()}, X: ", X.shape, type(X), X.type())
        return X


class Routeur(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(CONTEXT_LENGTH * EMBEDDING_DIM,
                             NUMBER_OF_BLOCKS + 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, X):
        printd(f"Lib->Routeur, l{lineno()}, X: ", X.shape, X.type())
        X = torch.flatten(X, start_dim=0)
        printd(f"Lib->Routeur, l{lineno()}, X: ", X.shape, X.type())
        X = self.lin(X)
        printd(f"Lib->Routeur, l{lineno()}, X: ", X.shape, X.type())
        X = self.softmax(X)
        printd(f"Lib->Routeur, l{lineno()}, X: ", X.shape, X.type())
        idx = torch.multinomial(X, 1)
        printd(f"Lib->Routeur, l{lineno()}, ids: ", idx, type(idx), idx.type())
        return idx


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(EMBEDDING_DIM, HIDDEN_DIM)
        self.gelu = nn.GELU()
        self.lin2 = nn.Linear(HIDDEN_DIM, EMBEDDING_DIM)

    def forward(self, X):
        printd(f"Lib->FeedForward, l{lineno()}, X : ", X.shape, X.type())
        X = self.lin1(X)
        printd(f"Lib->FeedForward, l{lineno()}, X : ", X.shape, X.type())
        X = self.gelu(X)
        printd(f"Lib->FeedForward, l{lineno()}, X : ", X.shape, X.type())
        X = self.lin2(X)
        printd(f"Lib->FeedForward, l{lineno()}, X : ", X.shape, X.type())
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

        printd(f"Lib->NextTkPrediction, l{lineno()}, X : ", X.shape, X.type())
        topk, indices = torch.topk(X, k)
        filtered_indices = indices
        if len(filtered_indices) == 0:
            return torch.multinomial(X, 1)
        else:
            return torch.multinomial(X[filtered_indices], 1)

    def forward(self, X, topk_sorted_index_limit=None):
        X = torch.flatten(X, start_dim=0)
        printd(f"Lib->NextTkPrediction, l{lineno()}, X : ", X.shape, X.type())
        X = self.lin(X)
        printd(f"Lib->NextTkPrediction, l{lineno()}, X : ", X.shape, X.type())
        X = self.softmax(X)
        printd(f"Lib->NextTkPrediction, l{lineno()}, X : ", X.shape, X.type())

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

        printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, X : ",
               X.shape, X.type())

        if (len(X.shape) == 2):
            _, _ = X.shape

            query = self.query_linear(X)
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, Query : ",
                   query.shape)
            query = query.view(CONTEXT_LENGTH,
                               NUMBER_OF_ATTENTION_HEADS,
                               HIDDEN_DIM // NUMBER_OF_ATTENTION_HEADS)
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, Query : ",
                   query.shape)
            key = self.key_linear(X)
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, Key : ",
                   key.shape)
            key = key.view(CONTEXT_LENGTH, NUMBER_OF_ATTENTION_HEADS,
                           HIDDEN_DIM // NUMBER_OF_ATTENTION_HEADS)
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, Key : ",
                   key.shape)
            value = self.value_linear(X)
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, Value : ",
                   value.shape)
            value = value.view(CONTEXT_LENGTH,
                               NUMBER_OF_ATTENTION_HEADS,
                               HIDDEN_DIM // NUMBER_OF_ATTENTION_HEADS)
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, Value : ",
                   value.shape)

            attention = torch.matmul(query,
                                     key.transpose(-1, -2)) / math.sqrt(
                                     HIDDEN_DIM // NUMBER_OF_ATTENTION_HEADS)
            attention = self.softmax(attention)

            output = torch.matmul(attention, value).view(CONTEXT_LENGTH,
                                                         HIDDEN_DIM)
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, out : ",
                   output.shape)
        else:
            batch_size, _, _ = X.shape

            query = self.query_linear(X)
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, Query : ",
                   query.shape)
            query = query.view(batch_size, CONTEXT_LENGTH,
                               NUMBER_OF_ATTENTION_HEADS,
                               HIDDEN_DIM // NUMBER_OF_ATTENTION_HEADS)
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, Query : ",
                   query.shape)
            key = self.key_linear(X)
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, Key : ",
                   key.shape)
            key = key.view(batch_size, CONTEXT_LENGTH,
                           NUMBER_OF_ATTENTION_HEADS,
                           HIDDEN_DIM // NUMBER_OF_ATTENTION_HEADS)
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, Key : ",
                   key.shape)
            value = self.value_linear(X)
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, Value : ",
                   value.shape)
            value = value.view(batch_size, CONTEXT_LENGTH,
                               NUMBER_OF_ATTENTION_HEADS,
                               HIDDEN_DIM // NUMBER_OF_ATTENTION_HEADS)
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, Value : ",
                   value.shape)

            attention = torch.matmul(query,
                                     key.transpose(-1, -2)) / math.sqrt(
                                     HIDDEN_DIM // NUMBER_OF_ATTENTION_HEADS)
            attention = self.softmax(attention)

            output = torch.matmul(attention, value).view(batch_size,
                                                         CONTEXT_LENGTH,
                                                         HIDDEN_DIM)
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, out : ",
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
        printd(f"Lib->Block, l{lineno()}, X : ", X.shape, X.type())
        y = self.attention(X)
        printd(f"Lib->Block, l{lineno()}, X : ", X.shape, X.type())
        X = self.ln1(X+y)
        printd(f"Lib->Block, l{lineno()}, X : ", X.shape, X.type())
        y = self.ff(X)
        printd(f"Lib->Block, l{lineno()}, X : ", X.shape, X.type())
        X = self.ln2(X+y)
        printd(f"Lib->Block, l{lineno()}, X : ", X.shape, X.type())
        return X
