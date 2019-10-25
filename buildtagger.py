# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        # with dimensionality hidden_dim.

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    # use torch library to save model parameters, hyperparameters, etc. to model_file
    torch.manual_seed(1)
    
    infile = open(train_file)
    sents = infile.readlines()
    trainingData = []
    wordIndex = {}
    tagIndex = {}
    windex = 0
    tindex = 0
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
                tindex += 1
        trainingData.append(([word[0] for word in line], [word[1] for word in line]))
    
    model = LSTMTagger(32, 32, len(wordIndex), len(tagIndex))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), 0.05)
    
    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    with torch.no_grad():
        inputs = prepare_sequence(trainingData[0][0], wordIndex)
        tag_scores = model(inputs)
        print(tag_scores)

    for epoch in range(10):  # again, normally you would NOT do 300 epochs, it is toy data
        print(epoch)
        theloss = 0
        for i in range(0, 1000):
            sentence = trainingData[i][0]
            tags = trainingData[i][1]
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepare_sequence(sentence, wordIndex)
            targets = prepare_sequence(tags, tagIndex)

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
        inputs = prepare_sequence(trainingData[0][0], wordIndex)
        tag_scores = model(inputs)

        # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
        # for word i. The predicted tag is the maximum scoring tag.
        # Here, we can see the predicted sequence below is 0 1 2 0 1
        # since 0 is index of the maximum value of row 1,
        # 1 is the index of maximum value of row 2, etc.
        # Which is DET NOUN VERB DET NOUN, the correct sequence!
        print(tag_scores)
    
    print('Finished...')

def prepare_sequence(seq, toIndex):
    indices = [toIndex[w] for w in seq]
    return torch.tensor(indices, dtype=torch.long)

if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
