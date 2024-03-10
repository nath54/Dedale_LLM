import os
import torch

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
        self._files: dict[str | int, dict[str, str | int | ]] = {}
        #
        self._size_streaming = size_streaming
        self._prompt_per_stream = prompt_per_stream
        self._randomized_access = randomized_access
        assert tokenizer is not None, "Warning: Tokenizer must not be None!"
        self._tokenizer = tokenizer
        self.padding_context_length = padding_context_length
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
        assert isinstance(data_filepath, str), f"{data_filepath} is not a str"
        assert os.path.exists(data_filepath), f"Path {data_filepath} not found"
        assert key not in self._data, f"Key already used : {key}"
        #
        self._files[key] = {
            "file": open(data_filepath, "r"),
            "streamed": 0,
            "accesses_left": 0,
            "eof": False
        }
        self._data[key] = torch.Tensor(0)

    def next_stream_data(self, key: str | int = "default"):
        pass

    def get_data(
        self,
        nb_prompts: int = 1,
        nb_batch: int = 1,
        key: str | int = "default"
    ) -> torch.Tensor:
        #
        if nb_batch == 1:
            pass
        