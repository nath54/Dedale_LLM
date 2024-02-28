from lib import Embedding, Routeur, NextTokenPrediction, Block
from lib import MAX_LOADED_BLOCKS, CONTEXT_LENGTH, BLOCKS_TO_REMOVE
from lib import printd, tokenizer

import torch
import torch.nn as nn
import os

PASSAGE_STOP = -1
PASSAGE_CONTINUE_WITH_ROUTEUR = -2

class MixtofExp(nn.Module):
    def __init__(
        self,
        max_length: int = 50,
        max_routeur_passages: int = 8,
        force_passage: list[int] = []
    ) -> None:
        """
        The `__init__` function initializes the attributes of the `MixtofExp`
        class, including the embedding layer, routing layer,
        next token prediction layer, and various parameters.

        :param max_length: The `max_length` parameter represents
        the maximum length of the input sequence.
        It is used to determine the size of the embedding layer in the model,
        defaults to 50
        (optional)
        """
        super().__init__()
        #
        self.embedding: Embedding = Embedding()
        self.routeur: Routeur = Routeur()
        self.next_token_prediction: NextTokenPrediction = NextTokenPrediction()
        #
        self.max_routeur_passages: int = max_routeur_passages
        self.force_passage: list[int] = force_passage

        self.max_blocks: int = MAX_LOADED_BLOCKS
        self.nb_blocks_to_remove: int = BLOCKS_TO_REMOVE
        self.blocks: dict = {}
        #
        self.max_length = max_length

    def forward_block(self, X: torch.Tensor, block_id: int) -> torch.Tensor:
        if block_id not in self.blocks:
            self.load_block(block_id)
        #
        self.blocks[block_id]["usage"] += 1
        printd("Model, line 63, Passing by block ", block_id)
        #
        printd("Model, line 65, X : ", X.shape, type(X), X.type())
        X.type(torch.float)
        X = self.blocks[block_id]["model"].forward(X)
        X.type(torch.float)
        return X

    def forward_routeur_passage(
        self,
        X: torch.Tensor,
        routeur_passages: int = 1
    ) -> torch.Tensor:
        block_id_t: torch.Tensor = self.routeur(X)
        block_id: int = block_id_t.item()
        printd("Model, line 54, block_id : ", block_id, type(block_id))
        while block_id != 0 or routeur_passages < self.max_routeur_passages:
            routeur_passages += 1
            #
            X = self.forward_block(X, block_id)
            #
            block_id_t: torch.Tensor = self.routeur(X)
            block_id: int = block_id_t.item()
        #
        return X

    def forward(self, X: torch.Tensor):
        """
        The forward function takes an input tensor,
        applies embedding and routing operations, passes the
        input through multiple blocks, and returns the predicted next token.

        @param X The parameter X is a torch.Tensor, which represents
        the input data for the forward pass of the model.

        @return the predicted token (tk) after passing through
        the model and blocks.
        """
        #
        X = self.embedding(X)
        printd("Model, line 49, X: ", X.shape, type(X), X.type())
        X.type(torch.float)
        #
        routeur_passages: int = 1
        #
        if self.force_passage != []:
            for block_id in self.force_passage:
                if block_id == PASSAGE_STOP:
                    break
                elif block_id == PASSAGE_CONTINUE_WITH_ROUTEUR:
                    self.forward_routeur_passage(X, routeur_passages)
                    break
                #
                routeur_passages += 1
                #
                X = self.forward_block(X, block_id)
        else:
            self.forward_routeur_passage(X)
        #
        tk = self.next_token_prediction(X)
        printd("Model, line 73, tk : ", tk, tk.shape, type(tk), tk.type())
        return tk

    def forward_txt(self, txt: str):
        #
        X: torch.Tensor = tokenizer.encode(txt, max_length=CONTEXT_LENGTH,
                                           padding="max_length",
                                           truncation=True,
                                           return_tensors="pt")
        printd("Model->forward_txt, line 82 : ", X.shape, type(X))
        #
        return self.forward(X)

    def use(self, txt: str):
        """
        The function takes an input X, passes it through a forward function,
        converts the output index to tokens using a tokenizer,
        and returns the tokens.

        :param X: The parameter X is the input data that you want
        to pass through the model. It could be a single example
        or a batch of examples, depending on the implementation of the model
        :return: the token corresponding to the index obtained from
        the forward pass of the model.
        """
        idx = self.next_token_prediction.get_next_token_idx(
                self.forward_txt(txt))
        idx = idx.item()
        tk = tokenizer.convert_ids_to_tokens(idx)
        return tk

    def load_weights(self, filename: str = "weights/moe.pt") -> None:
        """
        The function loads weights from a file and sets the model
        to evaluation mode.

        :param filename: The `filename` parameter is a string that represents
        the path and name of the
        file from which the weights will be loaded.
        By default, it is set to "weights/moe.pt", defaults
        to weights/moe.pt
        :type filename: str (optional)
        """
        if os.path.exists(filename):
            self.load_state_dict(torch.load(filename))
            self.eval()

    def save_weights(self, filename: str = "weights/moe.pt") -> None:
        """
        The function saves the weights of a model to a specified file.

        :param filename: The `filename` parameter is a string that specifies
        the name and path of the file where the weights will be saved.
        By default, it is set to "weights/moe.pt", which means the
        weights will be saved in a file named "moe.pt"
        in the "weights" directory, defaults to weights/moe.pt
        :type filename: str (optional)
        """
        torch.save(self.state_dict(), filename)
        #
        for id_block in self.blocks:
            self.save_block(id_block)

    def load_block(self, block_id: int) -> None:
        """
        The `load_block` function loads a block of weights from a file
        and adds it to a dictionary of blocks, while also managing
        the number of blocks to ensure it doesn't exceed a maximum limit.

        :param block_id: The `block_id` parameter is an integer
        that represents the ID of the block to be loaded
        :type block_id: int
        """
        #
        if len(self.blocks) >= self.max_blocks:
            while len(self.blocks) <= self.max_blocks-self.nb_blocks_to_remove:
                min_block = None
                for i in self.blocks:
                    if min_block is None or \
                       self.blocks[min_block]["usage"] \
                       > self.blocks[i]["usage"]:

                        min_block = i
                #
                if min_block is not None:
                    self.unload_block(min_block)
                else:
                    break
        #
        if os.path.exists(f"weights/block_{block_id}.pt"):
            block = Block()
            block.load_state_dict(torch.load(f"weights/block_{block_id}.pt"))
            block.eval()
            #
            self.blocks[block_id] = {
                "model": block,
                "usage": 0
            }
        else:
            block = Block()
            #
            self.blocks[block_id] = {
                "model": block,
                "usage": 0
            }

    def save_block(self, block_id: int) -> None:
        """
        The function saves the state dictionary of a model associated
        with a given block ID to a file.

        :param block_id: The `block_id` parameter is an integer
        that represents the unique identifier of
        a block
        :type block_id: int
        """
        if block_id in self.blocks:
            torch.save(self.blocks[block_id]["model"].state_dict(),
                       f"weights/block_{block_id}.pt")

    def unload_block(self, block_id: int) -> None:
        """
        The function unloads a block by saving it and removing it from
        the blocks dictionary.

        :param block_id: The `block_id` parameter is an integer
        that represents the ID of the block that
        needs to be unloaded
        :type block_id: int
        """
        assert block_id in self.blocks
        #
        self.save_block(block_id)
        del (self.blocks[block_id])
