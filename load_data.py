import pandas as pd
import json

file_path = "/home/nathan/Documents/Datasets/Language/Wikipedia/parts/wikipedia_part.000.jsonl"

with open(file_path) as f:
    df = pd.DataFrame(json.loads(line) for line in f)

print(df.values[0])
text = df.values[0][0]
title = df.values[0][1]["text"]
url = df.values[0][1]["url"]
language = df.values[0][1]["language"]
timestamp = df.values[0][1]["timestamp"]
