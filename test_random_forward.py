from lib import tokenizer, config
from model import MixtofExp

if __name__ == "__main__":
    model = MixtofExp(
        force_passage=config["force_passage"],
        model_name=config["model_name"]
    )
    model.load_weights()

    t1 = "1 2 3 4 5 6 7 8"

    c = 0
    mc = 5
    out = None
    while c < mc and out != tokenizer.eos_token:
        c += 1
        if len(t1) >= config["context_length"]:
            out = model.use(t1[-config["context_length"]:])
        else:
            out = model.use(t1)
        out = out.replace("Ġ", " ")
        t1 += out
        print(t1)
