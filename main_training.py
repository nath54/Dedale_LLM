from model import MixtofExp
from train import train
from lib import config

if __name__ == "__main__":
    model = MixtofExp(
        force_passage=config["force_passage"],
        model_name=config["model_name"]
    )
    model.load_weights()
    train(model)
    model.save_weights()
