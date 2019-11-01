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

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, char_size, char_embed_dim, convDim, wordLength):
        super(LSTMTagger, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.convDim = convDim
        
        self.cnn = CNN(char_size, char_embed_dim, convDim, wordLength)
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim + convDim, hidden_dim, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)

    def forward(self, characters, sentence):
        charTList = []
        for word in characters:
            result = self.cnn(word)
            result = result.unsqueeze(0)
            charTList.append(result)
        charTensors = torch.cat(charTList, dim=0)
        embeds = self.word_embeddings(sentence)
        embeds = torch.cat((embeds, charTensors), dim=1)
        
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_pred = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return tag_pred

class CNN(nn.Module):
    def __init__(self, char_size, char_embed_dim, convDim, wordLength):
        super(CNN, self).__init__()
        self.wordLength = wordLength
        self.char_embed_dim = char_embed_dim
        
        self.char_embeddings = nn.Embedding(char_size+1, char_embed_dim, char_size)
        self.conv1 = nn.Conv1d(1, convDim, kernel_size=3*char_embed_dim, stride=char_embed_dim, padding=char_embed_dim)
        self.pool = nn.MaxPool1d(kernel_size=wordLength, padding=0)
     
    def forward(self, word):
        embeds = self.char_embeddings(word)
        embeds = embeds.reshape(1, 1, -1)
        out = self.conv1(embeds)
        out = F.relu(out)
        out = out.reshape(1, 1, -1)
        out = self.pool(out)
        return out.view(-1)

def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    # use torch library to save model parameters, hyperparameters, etc. to model_file
    torch.manual_seed(1)
    maxWordLength = 20
    BATCH_SIZE = 20
    
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
            if (len(line[i][0]) > maxWordLength):
                maxWordLength = max(maxWordLength, len(line[i][0]))
                print(line[i][0])
        trainingData.append(([word[0] for word in line], [word[1] for word in line]))
    print(maxWordLength)
    model = LSTMTagger(24, 24, len(wordIndex), len(tagIndex), len(charIndex), 12, 12, maxWordLength)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), 0.1)
    print(maxWordLength)
    for epoch in range(1):
        theloss = 0
        print(str(epoch) + "-------")
        for i in range(0, len(trainingData)):
            sentence = trainingData[i][0]
            tags = trainingData[i][1]
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepareSequence(sentence, wordIndex)
            targets = prepareSequence(tags, tagIndex)
            charRep = prepareChars(sentence, charIndex, maxWordLength)

            # Step 3. Run our forward pass.
            tag_scores = model(charRep, sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
            theloss += loss.item()
            if (i % 1000 == 0):
                print(i, theloss)
                theloss = 0
            
    # See what the scores are after training
    with torch.no_grad():
        inputs = prepareSequence(trainingData[0][0], wordIndex)
        charRep = prepareChars(trainingData[0][0], charIndex, maxWordLength)
        tag_scores = model(charRep, inputs)

        scores, maxIndices = torch.max(tag_scores, 1)
        for i in range(0, len(trainingData[0][0])):
            print(trainingData[0][0][i] + "/" + reverseTagIndex[maxIndices[i].item()])
            
            
    #save to file
    torch.save((wordIndex, reverseTagIndex, charIndex, model.state_dict()), model_file)
    print('Finished...')

#take word or tag
def prepareSequence(seq, toIndex):
    indices = [toIndex[w] for w in seq]
    return torch.tensor(indices, dtype=torch.long)

def prepareChars(sentence, charIndex, maxWordLength):
    indices = []
    for w in sentence:
        cindex = []
        for i in range(0, len(w)):
            if w[i] in charIndex:
                cindex.append(charIndex[w[i]])
            else:
                cindex.append(len(charIndex))
        for i in range(len(w), maxWordLength):
            cindex.append(len(charIndex))
        indices.append(cindex)
    return torch.tensor(indices, dtype=torch.long)

if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
