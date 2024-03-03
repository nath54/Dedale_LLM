from model import MixtofExp
from train import train
from config import CONFIG_MODEL_NAME, CONFIG_MODEL_FORCE_PASSAGE

if __name__ == "__main__":
    model = MixtofExp(
        force_passage=CONFIG_MODEL_FORCE_PASSAGE,
        model_name=CONFIG_MODEL_NAME
    )
    model.load_weights()
    train(model, 100)
    model.save_weights()
