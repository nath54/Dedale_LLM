from lib import tokenizer
from model import MixtofExp
from config import CONFIG_MODEL_NAME, CONFIG_MODEL_FORCE_PASSAGE

if __name__ == "__main__":
    model = MixtofExp(
        force_passage=CONFIG_MODEL_FORCE_PASSAGE,
        model_name=CONFIG_MODEL_NAME
    )
    model.load_weights()

    t1 = "3 4 5 6 "

    c = 0
    mc = 5
    out = None
    while c < mc and out != tokenizer.eos_token:
        c += 1
        out = model.use(t1)
        out = out.replace("Ġ", " ")
        t1 += out
        print(t1)
