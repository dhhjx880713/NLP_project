# Imports

import numpy as np
import pandas as pd
import regex as re
import preprocessor as p
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"

device = torch.device(dev)

# data = pd.read_csv('data/hindi_hatespeech.tsv', sep='\t', encoding="utf8")
data = pd.read_csv('data/bengali_hatespeech.csv', encoding="utf8")
data = data.rename(columns={"sentence": "text"})
# Split off a small part of the corpus as a development set (~100 data points)
data = data.head(3000)
# Data preparation
p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.SMILEY)
# f = open('data/stopwords-hi.txt', 'r', encoding="utf8")
f = open('data/stopwords-bn.txt', 'r', encoding="utf8")
rules = f.read().splitlines()
f.close()

def getWords(text):
    # remove stop words
    for i in rules:
        temp = i.strip()
        text = re.sub(r'(\W|^){}(\W|$)'.format(temp), ' ', text, flags=re.I)

    text = p.clean(text)  # Remove URLs, mentions, and smileys.
    text = re.sub(r'['u'\U0001F600-\U0001F64F'']+', ' ', text, flags=re.UNICODE)  # Remove emoticons
    text = re.sub(r'#[^\s]+', '', text)  # Remove hashtags
    text = re.sub(r'\p{P}+', ' ', text)  # Remove punctuations
    text = text.lower()  # Lower letters
    return [i for i in re.compile('[^\d\W]+').findall(text) if i.isalpha() is False]

# Compile a list of all words in the development set

data['processed'] = data.text.apply(getWords)

data = data[data.processed.astype(str) != '[]']
C = [j for i in data.processed.to_list() for j in i]
V = list(set(C))

# Function word_to_one_hot
def word_to_one_hot(word):
    vector = np.zeros(vocab_size)
    vector[V.index(word)] = 1
    return vector

# Subsampling
def sampling_prob(word):
    z = C.count(word)/len(C)
    p = (np.sqrt(z/0.001)+1)*0.001/z
    return p

# Skip-Grams
def get_target_context(sentence):
    # The tokens from each piece of text are already in a list so I'd start from there
    # Sentences are from data.processed
    sentence = [i for i in sentence if sampling_prob(i)>np.random.sample()]    # Drop frequent words

    training_pairs = []
    for i in sentence:
        target_idx = sentence.index(i)
        smallest_idx = max(target_idx-window_size, 0)
        largest_idx = min(target_idx+window_size, len(sentence)-1)
        for j in range(smallest_idx, largest_idx+1):
            if j == target_idx:
                continue
            training_pairs.append([i, sentence[j]])

    return training_pairs


# Set hyperparameters
window_size = 5
embedding_size = 300
vocab_size = len(V)

# More hyperparameters
learning_rate = 0.001
epochs = 100
batch_size = 200
stop_criterion = 5000

instances = []
for _, i in data.iterrows():
    instances.extend(get_target_context(i.processed))   # Get input-output pairs

# Create training samples
input = np.zeros((len(instances), vocab_size))
output = np.zeros((len(instances), vocab_size))
for i in range(len(instances)):
    input[i] = word_to_one_hot(instances[i][0])
    output[i] = word_to_one_hot(instances[i][1])

dataset = TensorDataset(torch.Tensor(input), torch.Tensor(output))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Create model
class Word2Vec(nn.Module):
  def __init__(self):
    super(Word2Vec, self).__init__()
    self.input = nn.Linear(vocab_size, embedding_size, bias=False)
    self.output = nn.Linear(embedding_size, vocab_size, bias=False)
    self.input.weight.data.uniform_(-1, 1)
    self.output.weight.data.uniform_(-1, 1)

  def forward(self, one_hot):
    #one_hot = one_hot.to(device)
    embed = self.input(one_hot)
    context = F.log_softmax(self.output(embed), dim=-1)
    return context


model = Word2Vec()
model.to(device)
embeddings = model.input.weight.data
context_mappings = model.output.weight.data

# Define optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Define train procedure
def train(dataloader, model, optimizer, criterion):
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = criterion(pred, torch.max(y, 1)[1])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"accumulated loss: {total_loss:>7f}")
    return total_loss


# instances = []
# for _, i in data.iterrows():
#     instances.extend(get_target_context(i.processed))   # Get input-output pairs

# # Create training samples
# input = np.zeros((len(instances), vocab_size))
# output = np.zeros((len(instances), vocab_size))
# for i in range(len(instances)):
#     input[i] = word_to_one_hot(instances[i][0])
#     output[i] = word_to_one_hot(instances[i][1])

# dataset = TensorDataset(torch.Tensor(input), torch.Tensor(output))
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# load initial weights
# model.input.weight = nn.Parameter(embeddings)
# model.output.weight = nn.Parameter(context_mappings)
loss_results = []
print("Training started")
model.train()
for i in range(epochs):
    print('\nEPOCH {}/{} '.format(i + 1, epochs))
    loss = train(dataloader, model, optimizer, criterion)
    loss_results.append(loss)
    if loss < stop_criterion:
        break
print("Training finished")

torch.save(model.input.state_dict(), 'bengali_embeddings.pth')
with open('bengali_vocab.txt', 'w') as f:
    for item in V:
        f.write("%s\n" % item)

np.save('bengali_loss.npy', np.asarray(loss_results))
# torch.save(model.input.state_dict(), 'hindi_embeddings.pth')
# with open('hindi_vocab.txt', 'w') as f:
#     for item in V:
#         f.write("%s\n" % item)