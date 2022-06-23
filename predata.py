import pandas as pd
import json


df1 = pd.read_json('Software.json', lines=True)

input_document = []
for text in df1["reviewText"]:
    input_document.append(text)

df2 = pd.read_json('Prime_Pantry.json', lines=True)

for text in df2["reviewText"]:
    input_document.append(text)
    
print(input_document[0])
print(input_document[400])