import pandas as pd


with open(r'E:\workspace\project\data\stopwords-bn.txt', 'r', encoding="utf-8") as f:
    res = f.read()
print(res)
# data = pd.read_csv('data/hindi_hatespeech.tsv', sep='\t', encoding="utf8")
# print(data.head(10))