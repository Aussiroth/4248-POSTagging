# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime
#import numpy as np

OPENTAG = "<s>"
CLOSETAG = "</s>"
SUFFIX_LENGTH = 2

tags = ["IN", "DT", "NNP", "CD", "NN", "``", "''", "POS", "-LRB-", "VBN", "NNS", "VBP", ",", "CC", "-RRB-", "VBD", "RB",
"TO", ".", "VBZ", "NNPS", "PRP", "PRP$", "VB", "JJ", "MD", "VBG", "RBR", ":", "WP", "WDT", "JJR", "PDT", "RBS", "WRB",
"JJS", "$", "RP", "FW", "EX", "SYM", "#", "LS", "UH", "WP$", "PRP$", "PRP"]

def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    infile = open(train_file)
    modelfile = open(model_file, "w")
    #Counts tag occurances
    totalTags = 0
    tagCount = {}
    #Counts number of times a tag t1 follows a tag t0
    tagPairCount = {}
    #Counts number of times a word w0 is tagged t0
    tagWordCount = {}
    tagCapsCount = {}
    
    tagSuffixCount = {}
    tagHyphenCount = {}
    
    #holds the vocabulary
    words = {}
    
    tagCount["<s>"] = 0
    tagCount["</s>"] = 0
    tagPairCount["<s>"] = {}
    tagPairCount["</s>"] = {}
    for tag in tags:
        tagCount[tag] = 0
        tagPairCount[tag] = {}
        tagWordCount[tag] = {}
        tagSuffixCount[tag] = {}
        tagCapsCount[tag] = [0, 0, 0, 0]
        tagHyphenCount[tag] = [0, 0]
    
    #to calc several things
    #Calculate P(w1 | w0) -> needs to calc all bigram pairs C(w0, w1) and so also C(w0)
    #Calculate P(w | t) -> Need to count each tag, then for each tag the occurances of a word
    for line in infile:
        sentence = line.rstrip().split(" ");
        tagCount["<s>"] += 1
        for i in range(0, len(sentence)):
            word = sentence[i].rsplit("/", 1)
            if word[0] in words:
                words[word[0]] += 1
            else:
                words[word[0]] = 1

            
            preTag = "<s>"
            if i != 0:
                preTag = sentence[i-1].rsplit("/", 1)
                preTag = preTag[1]
                
            tagCount[word[1]] += 1
            
            if word[1] in tagPairCount[preTag]:
                tagPairCount[preTag][word[1]] += 1
            else:
                tagPairCount[preTag][word[1]] = 1
                
            if word[0] in tagWordCount[word[1]]:
                tagWordCount[word[1]][word[0]] += 1
            else:
                tagWordCount[word[1]][word[0]] = 1
            
            #calc probabilities related to capitalisation here
            startLetter = word[0][0]
            if startLetter >= 'A' and startLetter <= 'Z':
                tagCapsCount[word[1]][0] += 1
            elif startLetter >= 'a' and startLetter <= 'z':
                tagCapsCount[word[1]][1] += 1
            elif startLetter >= '0' and startLetter <= '9':
                tagCapsCount[word[1]][2] += 1
            else:
                tagCapsCount[word[1]][3] += 1
            
            #calc probabilities related to suffixes here
            if len(word[0]) < SUFFIX_LENGTH:
                suffix = word[0]
            else:
                suffix = word[0][len(word[0])-SUFFIX_LENGTH:]
            suffix = str.lower(suffix)
            if suffix in tagSuffixCount[word[1]]:
                tagSuffixCount[word[1]][suffix] += 1
            else:
                tagSuffixCount[word[1]][suffix] = 1
                
            if word[0].find("-") > 0:
                tagHyphenCount[word[1]][0] += 1
            else:
                tagHyphenCount[word[1]][1] += 1
        for k in tagCount:
            totalTags += tagCount[k]
        #finally do the </s>
        lastTag = sentence[len(sentence)-1].rsplit("/", 1)
        lastTag = lastTag[1]
        if lastTag not in tagPairCount:
            tagPairCount[lastTag] = {}
        if CLOSETAG in tagPairCount[lastTag]:
            tagPairCount[lastTag][CLOSETAG] += 1
        else:
            tagPairCount[lastTag][CLOSETAG] = 1
        tagCount["</s>"] += 1
    infile.close()
    
    #smooth capitalisation stats and turn to log probability
    for k in tagCapsCount:
        for i in range(0, 4):
            if tagCapsCount[k][i] == 0:
                tagCapsCount[k][i] += 1
            tagCapsCount[k][i] = math.log(tagCapsCount[k][i]/(tagCount[k]+4))
    
    #avoid 0 tag pairs, smooth out probabilities a bit
    langa1, langa2 = deleted_interpolation(tagCount, tagPairCount, totalTags)
    for k in tagPairCount:
        for k2 in tagCount:
            if k2 in tagPairCount[k]:
                tagPairCount[k][k2] = langa2 * tagPairCount[k][k2] / tagCount[k] + langa1 * tagCount[k2] / totalTags
            else:
                tagPairCount[k][k2] = langa1 * tagCount[k2] / totalTags
            tagPairCount[k][k2] = math.log(tagPairCount[k][k2])
    
    
    #update suffix log probability
    for tag in tagSuffixCount:
        for suffix in tagSuffixCount[tag]:
            tagSuffixCount[tag][suffix] = math.log(tagSuffixCount[tag][suffix]/(tagCount[tag]))
            
    #update hyphen log probability
    minHyphenProb = 0
    for tag in tagHyphenCount:
        if tagHyphenCount[tag][0] > 0:
            tagHyphenCount[tag][0] = math.log(tagHyphenCount[tag][0]/(tagHyphenCount[tag][0]+tagHyphenCount[tag][1]))
        else:
            tagHyphenCount[tag][0] = -10
        if tagHyphenCount[tag][1] > 0:
            tagHyphenCount[tag][1] = math.log(tagHyphenCount[tag][1]/(tagHyphenCount[tag][0]+tagHyphenCount[tag][1]))
        else:
            tagHyphenCount[tag][1] = -10
  
    
    #Write individual tag into
    modelfile.write(str(len(tagCount))+"\n")
    for k in tagCount:
        outline = str(k) + " "
        outline += str(tagCount[k])
        modelfile.write(outline + "\n")

    #write tag bigram info
    modelfile.write(str(len(tagPairCount))+"\n")
    for k in tagPairCount:
        modelfile.write(str(k) + " " + str(len(tagPairCount[k]))+"\n")
        for k2 in tagPairCount[k]:
            outline = str(k2) + " " + str(tagPairCount[k][k2])
            modelfile.write(outline + "\n")

    #write tag word pair info
    modelfile.write(str(len(tagWordCount))+"\n")
    for k in tagWordCount:
        modelfile.write(str(k) + " " + str(len(tagWordCount[k]))+"\n")
        for k2 in tagWordCount[k]:
            outline = str(k2) + " " + str(tagWordCount[k][k2])
            modelfile.write(outline + "\n")
            if k2 == "yen":
                print(k, k2, tagWordCount[k][k2])

    #write tag capitalisation info
    modelfile.write(str(len(tagCapsCount))+"\n")
    for k in tagCapsCount:
        outline = str(k) + " "
        sum = 0
        for i in range(0, len(tagCapsCount[k])):
            outline += str(tagCapsCount[k][i]) + " "
            sum += tagCapsCount[k][i]
        outline += str(sum)
        modelfile.write(outline + "\n")
    
    #write tag suffix pair info
    modelfile.write(str(len(tagSuffixCount)) + "\n")
    for k in tagSuffixCount:
        modelfile.write(str(k) + " " + str(len(tagSuffixCount[k])) + "\n")
        for suffix in tagSuffixCount[k]:
            outline = str(suffix) + " " + str(tagSuffixCount[k][suffix])
            modelfile.write(outline + "\n")

    #write hyphen info
    modelfile.write(str(len(tagHyphenCount)) + "\n")
    for tag in tagHyphenCount:
        modelfile.write(str(tag) + " " + str(tagHyphenCount[tag][0]) + " " + str(tagHyphenCount[tag][1]) + "\n")
        
    #write vocabulary
    modelfile.write(str(len(words)) + "\n")
    for word in words:
        modelfile.write(word + " " + str(words[word]) + "\n")
    modelfile.close()
    print('Finished...')

def deleted_interpolation(tagCount, tagPairCount, totalTags):
    langa1 = 0
    langa2 = 0
    for k in tagPairCount:
        for k2 in tagPairCount[k]:
            if tagCount[k] > 1:
                if k2 in tagPairCount[k]:
                    case1 = (tagPairCount[k][k2] - 1) / (tagCount[k] - 1)
                else:
                    case1 = 0
            else:
                case1 = 0
            case2 = (tagCount[k2] - 1) / (totalTags - 1)
            if case1 > case2:
                langa2 += tagPairCount[k][k2]
            else:
                langa1 += tagPairCount[k][k2]
    sum = langa1 + langa2
    return langa1/sum, langa2/sum
    
if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
