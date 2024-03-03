import pandas as pd
import json

BASE_FILE_PART = "/home/nathan/Documents/"
file_path = BASE_FILE_PART \
                + "Datasets/Language/Wikipedia/parts/wikipedia_part.000.jsonl"

with open(file_path) as f:
    df = pd.DataFrame(json.loads(line) for line in f)

print(df.values[0])
text = df.values[0][0]
title = df.values[0][1]["text"]
url = df.values[0][1]["url"]
language = df.values[0][1]["language"]
timestamp = df.values[0][1]["timestamp"]
