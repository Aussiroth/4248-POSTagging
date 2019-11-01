# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, char_size, char_embed_dim, conv_dim, word_length):
        super(LSTMTagger, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.conv_dim = conv_dim
        
        self.cnn = CNN(char_size, char_embed_dim, conv_dim, word_length)
        
        self.word_embeddings = nn.Embedding(vocab_size+1, embedding_dim, vocab_size)

        self.lstm = nn.LSTM(embedding_dim + conv_dim, hidden_dim, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)

    def forward(self, characters, sentence):
        charTensors = self.cnn(characters)
        embeds = self.word_embeddings(sentence)
        embeds = torch.cat((embeds, charTensors), dim=1)
        
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_pred = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return tag_pred

class CNN(nn.Module):
    def __init__(self, char_size, char_embed_dim, conv_dim, word_length):
        super(CNN, self).__init__()
        self.word_length = word_length
        self.char_embed_dim = char_embed_dim
        self.conv_dim = conv_dim
        
        self.char_embeddings = nn.Embedding(char_size+1, char_embed_dim, char_size)
        self.conv1 = nn.Conv1d(1, conv_dim, kernel_size=3*char_embed_dim, stride=char_embed_dim, padding=char_embed_dim)
        self.pool = nn.MaxPool1d(kernel_size=word_length, padding=0)
     
    def forward(self, word):
        embeds = self.char_embeddings(word)
        embeds = embeds.reshape(-1, 1, self.char_embed_dim*self.word_length)
        out = self.conv1(embeds)
        out = F.relu(out)
        out = out.reshape(-1, 1, self.conv_dim*self.word_length)
        out = self.pool(out)
        return out.view(-1, self.conv_dim)

def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    # use torch library to save model parameters, hyperparameters, etc. to model_file
    #torch.manual_seed(1)
    
    #device = torch.device('cpu')
    if (torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    maxWordLength = 20
    
    infile = open(train_file)
    sents = infile.readlines()
    trainingData = []
    wordIndex = {}
    tagIndex = {}
    charIndex = {}
    reverseTagIndex = {}
    windex = 0
    tindex = 0
    cindex = 0
    for line in sents:
        line = line.rstrip().split()
        for i in range(0, len(line)):
            line[i] = line[i].rsplit("/", 1)
            word = line[i]
            if word[0] not in wordIndex:
                wordIndex[word[0]] = windex
                windex += 1
            if word[1] not in tagIndex:
                tagIndex[word[1]] = tindex
                reverseTagIndex[tindex] = word[1]
                tindex += 1
            if (len(charIndex) < 1000):
                for char in word[0]:
                    if char not in charIndex:
                        charIndex[char] = cindex
                        cindex += 1
        trainingData.append(([word[0] for word in line], [word[1] for word in line]))

    model = LSTMTagger(16, 16, len(wordIndex), len(tagIndex), len(charIndex), 16, 16, maxWordLength).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), 0.1)

    for epoch in range(1):
        theloss = 0
        for i in range(0, len(trainingData)):
            sentence = trainingData[i][0]
            tags = trainingData[i][1]
            
            model.zero_grad()
            
            sentence_in = prepareSequence(sentence, wordIndex, device)
            targets = prepareSequence(tags, tagIndex, device)
            charRep = prepareChars(sentence, charIndex, maxWordLength, device)

            tag_scores = model(charRep, sentence_in)

            loss = loss_function(tag_scores, targets)
            loss.backward()
            theloss += loss.item()
            
            optimizer.step()
            
            if (i % 1000 == 0):
                print(i, theloss)
                theloss = 0
            
            
    #save to file
    torch.save((wordIndex, reverseTagIndex, charIndex, model.state_dict()), model_file)
    print('Finished...')

#take word or tag
def prepareSequence(seq, toIndex, device):
    indices = [toIndex[w] for w in seq]
    return torch.tensor(indices, dtype=torch.long).to(device)

def prepareWords(sentences, maxWords, wordIndex):
    result = []
    for i in range(0, maxWords):
        cindex = []
        for j in range(0, len(sentences)):
            if i < len(sentences[j]):
                cindex.append(wordIndex[sentences[j][i]])
            else:
                cindex.append(len(wordIndex))
        result.append(cindex)
    return result

def prepareChars(sentence, charIndex, maxWordLength, device):
    indices = []
    for w in sentence:
        cindex = []
        if (len(w) > maxWordLength):
            startPos = max(0, len(w) - maxWordLength)
        else:
            startPos = 0
        for i in range(startPos, len(w)):
            if w[i] in charIndex:
                cindex.append(charIndex[w[i]])
            else:
                cindex.append(len(charIndex))
        for i in range(len(w), maxWordLength):
            cindex.append(len(charIndex))
        indices.append(cindex)
    return torch.tensor(indices, dtype=torch.long).to(device)

if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
