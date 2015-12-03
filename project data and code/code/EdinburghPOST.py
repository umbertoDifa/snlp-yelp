import nltk
import json
from nltk.tokenize import TweetTokenizer
import copy

dataAllReview = []
dataReviewsPOS = []

with open('ReviewsOfbusinessEdinburgh.json') as inputFile:
    dataAllReview = json.load(inputFile)
    print('Load'+str(inputFile))
    inputFile.close()

tweetTk = TweetTokenizer()
tk = nltk.data.load('tokenizers/punkt/english.pickle')

i=0
len1 = len(dataAllReview)

for line in dataAllReview:
    i +=1
    print(str(i)+'/'+str(len1))
    newLine = copy.deepcopy(line)
    newLine['text'] = []
    for sentence in tk.tokenize(line['text']):
        x = nltk.pos_tag(tweetTk.tokenize(sentence))
        newLine['text'].append(x)
    dataReviewsPOS.append(newLine)

with open('ReviewsOfEdinburghPOS.json', 'w') as outputFile:
    print('Writing: '+str(outputFile))
    json.dump(dataReviewsPOS, outputFile)
    outputFile.close()
