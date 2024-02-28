from model import MixtofExp
from train import train

if __name__ == "__main__":
    model = MixtofExp(force_passage=[0, 1, 2])
    model.load_weights()
    train(model, 20)
    model.save_weights()
