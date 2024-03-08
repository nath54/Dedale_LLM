from lib import tokenizer, config
from model import MixtofExp

if __name__ == "__main__":
    model = MixtofExp(
        force_passage=config["force_passage"],
        model_name=config["model_name"]
    )
    model.load_weights()

    t1 = "31 32 33 34 35 36 37 38 39 40"

    c = 0
    mc = 50
    out = None
    while c < mc and out != tokenizer.eos_token:
        c += 1
        if len(t1) >= config["context_length"]:
            inp = t1[-config["context_length"]:]
            print("Input of the model : ", inp)
            out = model.use(inp)
        else:
            out = model.use(t1)
        out = out.replace("Ġ", " ")
        t1 += out
        print(t1)
