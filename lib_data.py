import os
import torch
from io import TextIOWrapper
from random import randint

# Theses are constants for the DataContainer class
# DataContainer Mode :
# Empty: No data in the DataContainer
DC_MODE_EMPTY: int = 0
# Single: DataContainer only contains one dataset
DC_MODE_SINGLE: int = 1
# DictAccess: DataContainer
DC_MODE_DICT_ACCESS: int = 2


###
# This class will contain the data which will be used to train a language model
# It will be able to contains separately differents training sets.
# It will store each sets in a different value of a dictionary, accessible with
# its associated key
# (that you will give when adding a dataset in this container)
###
class DataContainer:
    """_summary_
    If single file path, it will use the key 'default' in `self._data`.
    If list of files paths, it will use the index of each filepath as the
    access key.

    For optimisation, it will stream the data from the files, not loading each
    files completely. So it will contains the files pointers in `self._files`.
    It will streams `self._size_streaming` octets for `self._prompt_per_stream`
    asked prompts (for each dataset).

    If `self._randomized_access` is False, then the given prompts are given by
    slices in the order, elsewhere it will give random prompts.
    """
    def __init__(
        self,
        paths: None | str | list[str] | dict[str | int, str] = None,
        size_streaming: int = 10000,
        prompt_per_stream: int = 100,
        randomized_access: bool = True,
        tokenizer=None,
        padding_context_length: int = -1
    ) -> None:
        self._mode: int = DC_MODE_EMPTY
        self._data: dict[str | int, torch.Tensor] = {}
        self._files: dict[str | int, dict[str, bool | int | TextIOWrapper]] = {
        }
        #
        self._size_streaming = size_streaming
        self._prompt_per_stream = prompt_per_stream
        self._randomized_access = randomized_access
        #
        assert tokenizer is not None, "Warning: Tokenizer must not be None!"
        assert randomized_access or prompt_per_stream > size_streaming + 2
        #
        self._tokenizer = tokenizer
        self._padding_context_length = padding_context_length
        self._end_token: torch.Tensor = torch.Tensor(
            [self._tokenizer.eos_token])[0].to(int)
        #
        if isinstance(paths, str):
            self._mode = DC_MODE_SINGLE
            self.add_data('default', paths)
        elif isinstance(paths, dict):
            self._mode = DC_MODE_DICT_ACCESS
            for k in paths:
                self.add_data(k, paths[k])
        elif isinstance(paths, list[str]):
            self._mode = DC_MODE_DICT_ACCESS
            for i in range(paths):
                self.add_data(i, paths[i])
        elif paths is not None:
            raise UserWarning("Argument error, paths must be one of : \
[None, str, list[str], dict[key: str]]")

    #
    def add_data(
        self,
        key: str | int,
        data_filepath: str
    ) -> None:
        #
        assert isinstance(data_filepath, str), f"{data_filepath} is not a str"
        assert os.path.exists(data_filepath), f"Path {data_filepath} not found"
        assert key not in self._data, f"Key already used : {key}"
        #
        self._files[key] = {
            "file": open(data_filepath, "r"),
            "streamed": 0,
            "accesses_left": 0,
            "cursor": 0,
            "eof": False
        }
        self._data[key] = torch.Tensor(0)

    #
    def next_stream_data(self, key: str | int = "default") :
        #
        assert key in self._files, "Key error !"
        # If end of file, returning to the beginning
        if self._files[key]["eof"]:
            self._files[key]["file"].seek(0)
            self._files[key]["eof"] = False
        # Reading the file
        txt: str = self._files[key]["file"].read(self._size_streaming)
        # Test End of file
        if len(txt) < self._size_streaming:
            self._files[key]["eof"] = True
        # Tokenize the text
        self._data[key] = self._tokenizer.encode(
            txt,
            max_length = self.padding_context_length,
            padding = "max_length" if self._padding_context_length else "none",
            truncation = True,
            return_tensors = "pt"
        )
        #
        self._files[key]["cursor"] = 0
        self._files[key]["accesses_left"] = self._prompt_per_stream
    
    #
    def get_data_1batch(self,
        nb_prompts: int = 1,
        key: str | int = "default"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        #
        assert key in self._files, "Key error !"
        #
        X: torch.Tensor = torch.zeros(
            (nb_prompts, self._padding_context_length))
        Y: torch.Tensor = torch.zeros((nb_prompts, 1))
        #
        if self._randomized_access:
            len_data: int = self._data[key].size()[0]
            for k in range(nb_prompts):
                #
                if self._files[key]["accesses_left"] <= 0:
                    self.next_stream_data(key)
                #
                i: int = randint(0, len_data)
                s: int = min(self._padding_context_length, len_data-i)
                X[k, 0:s] = self._data[key][i:i+s]
                if i+s >= len_data - 1: # End of file
                    Y[k] = self._end_token
                else:
                    Y[k] = self._data[key][i+s]
                #
                self._files[key]["accesses_left"] -= 1
        else:
            len_data: int = self._data[key].size()[0]
            for k in range(nb_prompts):
                #
                if self._files[key]["accesses_left"] <= 0:
                    self.next_stream_data(key)
                #
                i: int = self._file[key]["cursor"]
                s: int = min(self._padding_context_length, len_data-i)
                X[k, 0:s] = self._data[key][i:i+s]
                if i+s >= len_data - 1: # End of file
                    Y[k] = self._end_token
                else:
                    Y[k] = self._data[key][i+s]
                #
                self._file[key]["cursor"] += 1
                self._files[key]["accesses_left"] -= 1
        #
        return (X, Y)
       
    #
    def get_data(
        self,
        nb_prompts: int = 1,
        nb_batch: int = 1,
        key: str | int = "default"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        #
        assert key in self._files, "Key error !"
        #
        if nb_batch == 1:
            return self.get_data_1batch(nb_prompts, key)
        else:
            X: torch.Tensor = torch.zeros(
                (nb_batch, nb_prompts, self._padding_context_length))
            Y: torch.Tensor = torch.zeros(
                (nb_batch, nb_prompts, 1))
            #
            for i in range(nb_batch):
                X[i], Y[i] = self.get_data_1batch(nb_prompts, key)
            #
            return (X, Y)
        