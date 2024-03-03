from torch import Tensor
import torch
from transformers import AutoTokenizer
import numpy as np

# Default token is the token
#  if the substring part hasn't any token correspondance
DEFAULT_TOKEN: int = 0

# NUMERIC_TOKENS makes the correspondance between its characters and tokens
CHARACTERS_TOKENS: str = " 0123456789.=+*/^<>()[]{}&~#'\"|%:,;\
                        abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\
                        \n\t"

CHARACTERS_TOKENS_VOCAB_SIZE: int = len(CHARACTERS_TOKENS)


class SingleCharactersTokenizer(AutoTokenizer):
    def __init__(self):
        # End of String token; indicates the end of a content.
        self.eos_token = DEFAULT_TOKEN

    """
    This function transforms the parameter `text_to_encode` into an array of
    tokens.

    @params:
        * text_to_encode: str = The string to encode into an array of tokens
        * max_length: int = The maximum length of an array of tokens. If its
            value is -1, this means that there is not any limits of length.
            Default value is -1. Elsewhere, it must be a positive integer.
        * padding: str = Indicates how the arrays of tokens should be padded.
            Possible values:
                - "none" (default): No padding
                - "max_length": If the array of tokens' size is less than
                    the parameter `max_length` (if != -1), it fills the array
                    with the value `DEFAULT_TOKEN`
        * truncation: bool = If True and if the parameter `max_length` is
            positive an the array of tokens' length is greater than
            `max_length`, then truncates the array of tokens to the length
            `max_lenght`. Default value is False.
        * return_tensor: str = Indicates the type of the returned result
            Possible values:
                - "lst" (default): returns a value of type list
                - "pt": returns a value of type torch.Tensor
                - "np": returns a value of type np.array
    """
    def encode(
            self,
            input_: str,
            max_length: int = -1,
            padding: str = "none",
            truncation: bool = False,
            return_tensors: str = "lst"
    ):
        # Verifying the arguments types
        assert isinstance(input_, str) or isinstance(input_, list[str]), \
               "Function Error: input_ must be str or list of str!"
        assert (max_length == -1 or max_length >= 0), \
               "Function Error: `max_length` must be equals to -1 or >= 0"
        assert padding in ["none", "max_length"], \
               "Function Error: `padding` must be one of [none, max_length]"
        assert return_tensors in ["lst", "pt", "np"], \
               "Function Error: `return_tensors` must be one of [lst, pt, np]"

        if isinstance(input_, str):
            input_ = [input_]

        # Initialising the array of tokens
        tokens: list[int] = []

        # For each text in input_, tokenize it
        for j in range(len(input_)):
            tokens.append([])
            # For each character of the text, assigning it to its token
            for i in range(len(input_[j])):
                if input_[j][i] in CHARACTERS_TOKENS:
                    # The + 1 shift is because the EOS_TOKEN is 0, so token 0
                    # is already took.
                    tk: int = 1 + CHARACTERS_TOKENS.index(input_[j][i])
                    tokens[j].append(tk)
                else:
                    tokens[j].append(DEFAULT_TOKEN)

        # Padding
        if padding == "max_length" and max_length >= 0:
            for j in range(len(tokens)):
                while len(tokens[j]) < max_length:
                    tokens[j].append(DEFAULT_TOKEN)

        # Return the array of tokens
        if return_tensors == "pt":
            return Tensor(tokens).to(torch.long)
        elif return_tensors == "np":
            return np.array(tokens, dtype=int)
        return tokens

    def get_vocab(self) -> list[str]:
        return [c for c in CHARACTERS_TOKENS]

    # Convert tokens to string
    def convert_ids_to_tokens(self, tokens_ids) -> str:
        #
        string_result: str = ""
        #
        tokens_id_lst: list[int] = []
        # Converts tokens_ids to list of integers
        if isinstance(tokens_ids, Tensor):
            tokens_id_lst = tokens_ids.to_list()
        elif isinstance(tokens_ids, np.ndarray):
            tokens_id_lst = tokens_ids.tolist()
        elif isinstance(tokens_ids, int):
            tokens_id_lst = [tokens_ids]
        else:
            tokens_id_lst = tokens_ids
        #
        for tk in tokens_id_lst:
            assert tk <= CHARACTERS_TOKENS_VOCAB_SIZE, "Error, bad token"
            #
            string_result += CHARACTERS_TOKENS[tk - 1]
        #
        return string_result
