
# import transformers
import torch
import torch.nn as nn
from torch import Tensor
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# from torch.utils.data import RandomSampler
import torch.optim as optim
# import torch.nn.functional as F
# from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib import device, tokenizer, CONTEXT_LENGTH


class Data:
    def __init__(self):
        self.paragraphs: list[str] = []
        self.paragraphs_tks: list[Tensor] = []
        #
        self.genere_data()
        #
        self.p: int = 0
        self.i: int = 0

    def genere_data(self):
        """
        The function "genere_data" appends numbers from 0 to 999
        to the first element of the "paragraphs" list.
        """
        self.paragraphs.append("")
        for x in range(1000):
            self.paragraphs[0] += str(x) + " "

        #

        for p in self.paragraphs:
            self.paragraphs_tks.append(tokenizer.encode(p,
                                                        return_tensors="pt"))

    def prepare_next_training_batch(self,
                                    context_length: int = CONTEXT_LENGTH,
                                    nb_batch: int = 1):
        """
        The function prepares the next training batch by encoding paragraphs
        using a tokenizer and returning the encoded tensors.

        @param context_length The `context_length` parameter represents
        the length of the context or
        sequence that you want to encode using the tokenizer.
        It determines how many tokens from the
        input text will be included in the encoded sequence.
        @param nb_batch The `nb_batch` parameter represents
        the number of training batches you want to
        prepare.
        It determines how many batches of training data will be created.
        """
        #
        if nb_batch == 1:
            # Single Batch
            X: Tensor = torch.zeros((context_length, ), dtype=torch.int)
            Y: Tensor = torch.zeros((1,), dtype=torch.int)
            a: int = max(0, self.i+1-context_length)
            b: int = self.i+1
            #
            c = 0
            for k in range(a, b):
                # print("DEBUG X : ", X, X.size())
                # print("DEBUG X[c] : ", X[c], X[c].size())
                # print("DEBUG self.paragraphs_tks[self.p][0, k] : ",
                #       self.paragraphs_tks[self.p][0, k],
                #       self.paragraphs_tks[self.p][0, k].size())
                X[c] = self.paragraphs_tks[self.p][0, k]
                c += 1
            while c < context_length:
                X[c] = 0  # tokenizer.eos_token
                c += 1
            #
            if (self.i+1 >= len(self.paragraphs[self.p])):
                Y = tokenizer.eos_token
                self.i = 0
                self.p = (self.p + 1) % len(self.paragraphs)
            else:
                Y = self.paragraphs_tks[self.p][0, self.i+1]
                self.i += 1
        else:
            # Multiple Batch
            X: Tensor = torch.zeros((nb_batch, context_length),
                                    dtype=torch.int)
            Y: Tensor = torch.zeros((nb_batch, 1), dtype=torch.int)
            for j in range(0, nb_batch):
                a: int = max(0, self.i+1-context_length)
                b: int = self.i+1
                t: Tensor = torch.zeros((context_length, ))
                #
                c = 0
                for k in range(a, b):
                    t[c] = self.paragraphs_tks[self.p][0, k]
                    c += 1
                while c < context_length:
                    t[c] = 0  # tokenizer.eos_token
                    c += 1
                #
                X[j] = t
                if (self.i+1 >= len(self.paragraphs[self.p])):
                    Y[j] = tokenizer.eos_token
                    self.i = 0
                    self.p = (self.p + 1) % len(self.paragraphs)
                else:
                    Y[j, 0] = self.paragraphs_tks[self.p][0, self.i+1]
                    self.i += 1
        #
        return X, Y


def train(model, epochs):
    #
    batch_size = 1
    #
    data: Data = Data()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss().to(device)
    #
    print("Preparing the model to train...")
    model.to(device)
    model.train()
    #
    tb = SummaryWriter()

    for epoch in range(epochs):

        losses_epoch = []

        print("epoch : ", epoch)

        training_size = len(data.paragraphs_tks[0])
        training_size = 1000
        loop = tqdm(range(training_size), leave=False, total=training_size)

        for i in loop:

            X, Y = data.prepare_next_training_batch(
                context_length=CONTEXT_LENGTH, nb_batch=batch_size)
            X.to(device)
            Y.to(device)

            optimizer.zero_grad()

            output = model(X).to(device)

            # print("Y : ", Y, Y.type(), ", output : ", output, output.type(),
            #       output.shape)
            loss = loss_fn(output, Y)
            loss.backward()
            losses_epoch.append(loss.item())

            optimizer.step()

            # Show progress while training
            loop.set_description(f'Epoch={epoch}/{epochs}')
            loop.set_postfix(loss=loss.item())

        tb.add_scalar("Loss", sum(losses_epoch)/len(losses_epoch), epoch)

        # for name, weight in model.classifier.named_parameters():
        #     tb.add_histogram(name,weight, epoch)
        #     tb.add_histogram(f'{name}.grad',weight.grad, epoch)

        model.save_weights("weights/weights.pt")
