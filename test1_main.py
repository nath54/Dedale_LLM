from model import MixtofExp
from train import train

if __name__ == "__main__":
    model = MixtofExp()
    train(model, 10)
