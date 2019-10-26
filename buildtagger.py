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

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

class CNNCharEmbed(nn.Module):

    def __init__(self, char_size, embedding_dim, convDim, ksize):
        super(CNNCharEmbed, self).__init__()
        
        self.char_embeddings = nn.Embedding(char_size, embedding_dim)
        
        #Input channels = 3, output channels = 18
        self.conv1 = nn.Conv1d(1, convDim, kernel_size = ksize, stride=1, padding=(ksize-1)/2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        #4608 input features, 64 output features (see sizing flow below)
        self.fc1 = nn.Linear(convDim * embedding_dim/2 * embedding_dim/2, 32)
        
        #Make final output vector equal to word embeeding vector length?
        self.fc2 = torch.nn.Linear(32, 16)
     
    def forward(self, word):
        embeds = self.word_embeddings(word)
        

def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    # use torch library to save model parameters, hyperparameters, etc. to model_file
    torch.manual_seed(1)
    
    infile = open(train_file)
    sents = infile.readlines()
    trainingData = []
    wordIndex = {}
    tagIndex = {}
    letterIndex = {}
    reverseTagIndex = {}
    windex = 0
    tindex = 0
    lindex = 0
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
        trainingData.append(([word[0] for word in line], [word[1] for word in line]))
    
    model = LSTMTagger(16, 16, len(wordIndex), len(tagIndex))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), 0.05)

    for epoch in range(1):
        print(epoch)
        model.zero_grad()
        for i in range(0, len(trainingData)):
            sentence = trainingData[i][0]
            tags = trainingData[i][1]
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepareSequence(sentence, wordIndex)
            targets = prepareSequence(tags, tagIndex)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
            theloss += loss.item()
        print(theloss)

    # See what the scores are after training
    with torch.no_grad():
        inputs = prepareSequence(trainingData[0][0], wordIndex)
        tag_scores = model(inputs)

        scores, maxIndices = torch.max(tag_scores, 1)
        for i in range(0, len(trainingData[0][0])):
            print(trainingData[0][0][i] + "/" + reverseTagIndex[maxIndices[i].item()])
    
    print('Finished...')

#take word 
def prepareSequence(seq, toIndex):
    indices = [toIndex[w] for w in seq]
    return torch.tensor(indices, dtype=torch.long)

if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
