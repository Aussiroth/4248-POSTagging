# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime
import string
#import numpy as np

#this will force ordering (and mapping to indexes) of tags when using the viterbi algorithm in the HMM

tags = ["IN", "DT", "NNP", "CD", "NN", "``", "''", "POS", "-LRB-", "VBN", "NNS", "VBP", ",", "CC", "-RRB-", "VBD", "RB",
"TO", ".", "VBZ", "NNPS", "PRP", "PRP$", "VB", "JJ", "MD", "VBG", "RBR", ":", "WP", "WDT", "JJR", "PDT", "RBS", "WRB",
"JJS", "$", "RP", "FW", "EX", "SYM", "#", "LS", "UH", "WP$"]

SUFFIX_LENGTH = 2

def tag_sentence(test_file, model_file, out_file):
    unknown_words = 0
    # write your code here. You can add functions as well.
    testfile = open(test_file)
    modelfile = open(model_file)
    outfile = open(out_file, "w")
    
    #tag counts
    totalTags = 0
    tagCounts = {}
    numtags = int(modelfile.readline().strip())
    for i in range(0, numtags):
        tagLine = modelfile.readline().rstrip().split()
        tagCounts[tagLine[0]] = int(tagLine[1])
        totalTags += int(tagLine[1])
    
    #tag pair counts in t0, t1 form
    tagPairCounts = {}
    numFirstTags = int(modelfile.readline().strip())
    for i in range(0, numFirstTags):
        firstTagLine = modelfile.readline().strip().split()
        firstTag = firstTagLine[0]
        tagPairCounts[firstTag] = {}
        numSecondTags = int(firstTagLine[1])
        for j in range(0, numSecondTags):
            tagLine = modelfile.readline().rstrip().split()
            tagPairCounts[firstTag][tagLine[0]] = float(tagLine[1])
    
    #read tag word pairs
    tagWordCounts = {}
    numTags = int(modelfile.readline().strip())
    for i in range(0, numTags):
        tagWordLine = modelfile.readline().strip().split()
        firstTag = tagWordLine[0]
        tagWordCounts[firstTag] = {}
        numWords = int(tagWordLine[1])
        for j in range(0, numWords):
            wordLine = modelfile.readline().strip().split()
            tagWordCounts[firstTag][wordLine[0]] = int(wordLine[1])
    
    #tag capitalisation probabilities
    tagCapsCounts = {}
    numtags = int(modelfile.readline().strip())
    for i in range(0, numtags):
        capLine = modelfile.readline().rstrip().split()
        tagCapsCounts[capLine[0]] = [float(capLine[i]) for i in range(1, len(capLine))]
    
    #tag suffix info
    tagSuffixCounts = {}
    numTags = int(modelfile.readline().strip())
    for i in range(0, numTags):
        tagSuffixLine = modelfile.readline().strip().split()
        firstTag = tagSuffixLine[0]
        tagSuffixCounts[firstTag] = {}
        for j in range(0, int(tagSuffixLine[1])):
            suffixLine = modelfile.readline().strip().split()
            tagSuffixCounts[firstTag][suffixLine[0]] = float(suffixLine[1])
    
    #tag hyphen info
    tagHyphenCounts = {}
    numTags = int(modelfile.readline().strip())
    for i in range(0, numTags):
        hyphenLine = modelfile.readline().strip().split()
        tagHyphenCounts[hyphenLine[0]] = [float(hyphenLine[i]) for i in range(1, len(hyphenLine))]
    
    #read words
    allWords = {}
    numWords = int(modelfile.readline().strip())
    for i in range(0, numWords):
        wordline = modelfile.readline().strip().split()
        allWords[wordline[0]] = int(wordline[1])
        
    for line in testfile:
        sentence = line.rstrip().split()
        #bookkeeping
        #best series of tags at each step for each node
        #best probability at each node
        probabilities = [10**50 for i in range(0, len(tags))]
        bestSequence = [[] for i in range(0, len(tags))]
        #now we perform viterbi
        #initialise leftmost variables
        #use log probability
        for i in range(0, len(tags)):
            cTag = tags[i]
            #perform p(tag | <s>)
            cProb = -tagPairCounts["<s>"][cTag]
            #do p(word | tag)
            if sentence[0] in allWords:
                if sentence[0] in tagWordCounts[cTag]:
                    cProb += -math.log(tagWordCounts[cTag][sentence[0]]/(tagCounts[cTag]+len(tagWordCounts[cTag])))
                else:
                    cProb += 10**2
            else:
                #capitalisation information is essentially lost in start of sentence
                if len(sentence[0]) > SUFFIX_LENGTH:
                    suffix = sentence[0][len(sentence[0])-SUFFIX_LENGTH:]
                else:
                    suffix = sentence[0]
                suffix = str.lower(suffix)
                if suffix in tagSuffixCounts[cTag]:
                    cProb += -tagSuffixCounts[cTag][suffix]
                else:
                    cProb += 15
                if sentence[0].find("-") > 0:
                    cProb += -tagHyphenCounts[cTag][0]
                else:
                    cProb += -tagHyphenCounts[cTag][1]
            probabilities[i] = cProb
            bestSequence[i].append(cTag)
        maxP = min(probabilities)
        #now perform the intermediate nodes
        for i in range(1, len(sentence)):
            newProbabilities = [10**50 for i in range(0, len(tags))]
            newBestSequence = [[] for i in range(0, len(tags))]
            for j in range(0, len(tags)):
                cTag = tags[j]
                #do p(word|tag)
                baseProb = 0
                if sentence[i] in allWords:
                    if sentence[i] in tagWordCounts[cTag]:
                        baseProb += -math.log(tagWordCounts[cTag][sentence[i]]/(tagCounts[cTag]+len(tagWordCounts[cTag])))
                    else:
                        baseProb += 10**2
                else:
                    #if word is not seen before, then make use of capitalisation and suffix
                    if len(sentence[i]) > SUFFIX_LENGTH:
                        suffix = sentence[i][len(sentence[i])-SUFFIX_LENGTH:]
                    else:
                        suffix = sentence[i]
                    suffix = str.lower(suffix)
                    if suffix in tagSuffixCounts[cTag]:
                        baseProb += -tagSuffixCounts[cTag][suffix]
                    else:
                        baseProb += 15
                    if sentence[i][0] >= 'A' and sentence[i][0] <= 'Z':
                        pos = 0
                    elif sentence[i][0] >= 'a' and sentence[i][0] <= 'z':
                        pos = 1
                    elif sentence[i][0] >= '0' and sentence[i][0] <= '9':
                        pos = 2
                    else:
                        pos = 3
                    baseProb +=  -tagCapsCounts[cTag][pos]
                    if sentence[i].find("-") > 0:
                        baseProb += -tagHyphenCounts[cTag][0]
                    else:
                        baseProb += -tagHyphenCounts[cTag][1]
                for k in range(0, len(tags)):
                    #perform p(tag | prev tag)
                    cProb = probabilities[k] - tagPairCounts[tags[k]][cTag]
                    cProb += baseProb
                    if cProb < newProbabilities[j]:
                        newProbabilities[j] = cProb
                        newBestSequence[j] = bestSequence[k].copy()
                        newBestSequence[j].append(cTag)
            probabilities = newProbabilities
            bestSequence = newBestSequence
            maxP = min(probabilities)
                
        #finally do the </s> for the viterbi algorithm
        finalBestProb = 10**50
        finalBestSol = []
        for i in range(0, len(tags)):
            cTag = "</s>"
            cProb = probabilities[i] - tagPairCounts[tags[i]][cTag]
            if cProb < finalBestProb:
                finalBestProb = cProb
                finalBestSol = bestSequence[i]
        outline = ""
        for i in range(0, len(sentence)):
            outline += sentence[i] + "/" + finalBestSol[i] + " "
        outline += "\n"
        outfile.write(outline)
    
    testfile.close()
    modelfile.close()
    outfile.close()
    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
