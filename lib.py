import torch
from torch import nn, Tensor
from transformers import AutoTokenizer
from custom_tokenisers import SingleCharactersTokenizer
import math
from inspect import currentframe
import sys
import json
import os


# Loading config
def test_config_file(config: dict) -> None:
    pass


# default config path (for later...)
config_path: str = "config.json"

if len(sys.argv) < 1:
    print("\u001b[31mError: no config file detected!\u001b[m")
    exit(1)
else:
    config_path = sys.argv[1]

print(f"Using config file: {config_path}")
if not config_path.endswith(".json"):
    print("\u001b[31mError: config file is not json!\u001b[m")
    exit(1)
#
if not os.path.exists(config_path):
    print("\u001b[31mError: config file doesn't exists!\u001b[m")
    exit(1)

config = {}
with open(config_path, "r") as f:
    config = json.load(f)

test_config_file(config)

#
if config["tokenizer"] == "falcon_tokenizer":
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b-instruct")
elif config["tokenizer"] == "single_characters_tokenizer":
    tokenizer = SingleCharactersTokenizer()
else:
    print("\u001b[31mError: bad value of `CONFIG_TOKENIZER`.\u001b[m")
    exit()

tokenizer.pad_token = tokenizer.eos_token
vocab = tokenizer.get_vocab()
VOCAB_SIZE = len(vocab)
print("Vocab size : ", VOCAB_SIZE)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


def lineno() -> int:
    cf = currentframe()
    return cf.f_back.f_lineno


def printd(*args):
    if config["print_dims"]:
        print("\u001b[34m * ", *args, "\u001b[m")


def print_params(model: nn.Module):
    for params in model.parameters():
        print(
            "\u001b[35m"
            f"{params}"
            "\u001b[m"
        )


class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, config["embedding_dim"])

    def forward(self, X):
        printd(f"Lib->Embedding, l{lineno()}, X: ", X.shape, X.type())
        X = self.embedding(X)
        printd(f"Lib->Embedding, l{lineno()}, X: ", X.shape, X.type())
        return X


class Routeur(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(
            config["context_length"] * config["embedding_dim"],
            config["nb_of_blocks"] + 1
        )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, X):
        printd(f"Lib->Routeur, l{lineno()}, X: ", X.shape, X.type())
        X = torch.flatten(X, start_dim=0)
        printd(f"Lib->Routeur, l{lineno()}, X: ", X.shape, X.type())
        X = self.lin(X)
        printd(f"Lib->Routeur, l{lineno()}, X: ", X.shape, X.type())
        X = self.softmax(X)
        printd(f"Lib->Routeur, l{lineno()}, X: ", X.shape, X.type())
        idx = torch.argmax(X, 1)
        printd(f"Lib->Routeur, l{lineno()}, ids: ", idx, type(idx), idx.type())
        return idx


class FeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        #
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        #
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.lin2 = nn.Linear(hidden_dim, output_dim)

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
        self.lin = nn.Linear(
            config["embedding_dim"]
            * config["context_length"],
            VOCAB_SIZE
        )
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
            return torch.argmax(X)
        else:
            return torch.argmax(X[filtered_indices])

    def forward(self, X, topk_sorted_index_limit=None):
        #
        if len(X.shape) >= 3:  # Si batch
            X = torch.flatten(X, start_dim=1)
        else:  # Pas de Batch
            X = torch.flatten(X, start_dim=0)
        #
        printd(f"Lib->NextTkPrediction, l{lineno()}, X : ", X.shape, X.type())
        X = self.lin(X)
        printd(f"Lib->NextTkPrediction, l{lineno()}, X : ", X.shape, X.type())

        return X

    def get_next_token_idx(self, X: Tensor, topk_sorted_index_limit=None):
        if topk_sorted_index_limit is int:
            idx = self.get_next_token_index(X, topk_sorted_index_limit)
        else:
            idx = torch.argmax(X)

        return idx


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, nb_of_attention_heads: int, hidden_dim: int):
        super().__init__()
        self.nb_of_attention_heads: int = nb_of_attention_heads
        self.hidden_dim: int = hidden_dim

        self.query_linear = nn.Linear(
            self.hidden_dim,
            self.hidden_dim
        ).to(device)
        # * self.nb_of_attention_heads)
        self.key_linear = nn.Linear(
            self.hidden_dim,
            self.hidden_dim
        ).to(device)
        # * self.nb_of_attention_heads)
        self.value_linear = nn.Linear(
            self.hidden_dim,
            self.hidden_dim
        ).to(device)
        # * self.nb_of_attention_heads)

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
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()},"
                   f"Passing without batchs.")

            query = self.query_linear(X)
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, Query : ",
                   query.shape)
            query = query.view(
                config["context_length"],
                self.nb_of_attention_heads,
                self.hidden_dim // self.nb_of_attention_heads
            )
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, Query : ",
                   query.shape)
            key = self.key_linear(X)
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, Key : ",
                   key.shape)
            key = key.view(
                config["context_length"],
                self.nb_of_attention_heads,
                self.hidden_dim // self.nb_of_attention_heads
            )
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, Key : ",
                   key.shape)
            value = self.value_linear(X)
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, Value : ",
                   value.shape)
            value = value.view(
                config["context_length"],
                self.nb_of_attention_heads,
                self.hidden_dim // self.nb_of_attention_heads
            )
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, Value : ",
                   value.shape)

            attention = torch.matmul(
                query,
                key.transpose(-1, -2)) / math.sqrt(
                    self.hidden_dim // self.nb_of_attention_heads
                )

            attention = self.softmax(attention)

            output = torch.matmul(attention, value).view(
                config["context_length"],
                self.hidden_dim
            )
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, out : ",
                   output.shape)
        else:
            # X = X.view(X.shape[0], X.shape[2], X.shape[3])
            # print(X.shape)
            batch_size, _, _ = X.shape

            printd(f"Lib->MultiHeadSelfAttention, l{lineno()},"
                   f"Passing with batchs : {batch_size}")

            printd(f"Lib->MultiHeadSelfAttention, l{lineno()},"
                   f" QueryLinear : {type(self.query_linear)}, "
                   f"{next(self.query_linear.parameters()).device}")

            query = self.query_linear(X)
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, Query : ",
                   query.shape)
            query = query.view(
                batch_size,
                config["context_length"],
                self.nb_of_attention_heads,
                self.hidden_dim // self.nb_of_attention_heads
            )
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, Query : ",
                   query.shape)
            key = self.key_linear(X)
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, Key : ",
                   key.shape)
            key = key.view(
                batch_size,
                config["context_length"],
                self.nb_of_attention_heads,
                self.hidden_dim // self.nb_of_attention_heads
            )
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, Key : ",
                   key.shape)
            value = self.value_linear(X)
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, Value : ",
                   value.shape)
            value = value.view(
                batch_size,
                config["context_length"],
                self.nb_of_attention_heads,
                self.hidden_dim // self.nb_of_attention_heads
            )
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, Value : ",
                   value.shape)

            attention = torch.matmul(
                query,
                key.transpose(-1, -2)
            ) / math.sqrt(
                self.hidden_dim // self.nb_of_attention_heads
            )
            attention = self.softmax(attention)

            output = torch.matmul(attention, value).view(
                batch_size,
                config["context_length"],
                self.hidden_dim
            )
            printd(f"Lib->MultiHeadSelfAttention, l{lineno()}, out : ",
                   output.shape)

        return output


class Block(nn.Module):
    def __init__(self, id_block: int = -1):
        super().__init__()
        self.id_block: int = id_block
        if (self.id_block == -1) or (self.id_block not in config[""].keys()):
            self.hidden_dim = config["blocks"]["default"]["hidden_dim"]
            self.nb_of_attention_heads = \
                config["blocks"]["default"]["nb_of_attention_heads"]
        else:
            self.hidden_dim = config["blocks"][self.id_block]["hidden_dim"]
            self.nb_of_attention_heads = \
                config["blocks"][self.id_block]["nb_of_attention_heads"]
        #
        self.attention = MultiHeadSelfAttention(
            nb_of_attention_heads=self.nb_of_attention_heads,
            hidden_dim=self.hidden_dim
        ).to(device)
        self.ln1 = nn.LayerNorm((config["embedding_dim"])).to(device)
        self.ff = FeedForward(
            input_dim=config["embedding_dim"],
            hidden_dim=self.hidden_dim,
            output_dim=config["embedding_dim"]
        ).to(device)
        self.ln2 = nn.LayerNorm((config["embedding_dim"])).to(device)

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
