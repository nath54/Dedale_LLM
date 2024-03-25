from model import MixtofExp
from train import train
from lib import config, print_params

if __name__ == "__main__":
    model = MixtofExp(
        force_passage=config["force_passage"],
        model_name=config["model_name"]
    )
    model.load_weights()
    #
    print_params(model)
    #
    train(model)
    model.save_weights()
