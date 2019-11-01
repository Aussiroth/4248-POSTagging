# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import torch

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

def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
	# use torch library to load model_file
    
    infile = open(test_file)
    outfile = open(out_file)
    
    wordIndex = {}
    tagIndex = {}
    charIndex = {}
    
    model = LSTMTagger(24, 24, len(wordIndex), len(tagIndex), len(charIndex), 12, 12, maxWordLength)
    
    wordIndex, tagIndex, charIndex, model_state_dict = torch.load(model_file)
    model.load_state_dict(model_state_dict)
    
    infile = open(test_file)
    sents = infile.readlines()
    for line in sents:
        line = line.rstrip().split()
        words = prepareWords(line, wordIndex)
        chars = prepareChars(line, charIndex, 20)
        with torch.no_grad():
            tag_scores = model(charRep, inputs)
        _, maxIndices = torch.max(tag_scores, 1)
        for i in range(0, len(line)):
            outline += line[i] + "/" + tagIndex[maxIndices[i].item()] + " "
        outline += "\n"
        outfile.write(outline)  
    infile.close()
    outfile.close()
    print('Finished...')

def prepareWords(word, wordIndex):
    indices = [wordIndex[w] for w in word]
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
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
