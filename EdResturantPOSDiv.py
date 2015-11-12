__author__ = 'vittorioselo'

import json
import nltk
from nltk.tokenize import TweetTokenizer
import copy



dataBusiness = []
dataReviews = []
dataReviewsPOS = []

dataRestReviews = []

dictRestBusiness = {}


with open('Dataset/BusinessByCity/businessEdinburgh.json') as inputFile:
    dataBusiness = json.load(inputFile)
    print('Load'+str(inputFile))
    inputFile.close()
# Restaurants
for line in dataBusiness:
    for x in line['categories']:
        if(x == 'Restaurants'):
            dictRestBusiness[line['business_id']] = x
print('Dictionary Resutrant Edinburgh Created!')

with open('Dataset/reviewsByCity/English/ReviewsOfbusinessEdinburgh.json') as inputFile:
    dataReviews = json.load(inputFile)
    print('Load'+str(inputFile))
    inputFile.close()

for line in dataReviews:
    if(dictRestBusiness.has_key(line['business_id'])):
        dataRestReviews.append(line)

with open('Dataset/ReviewsOfResutrantsEdinburgh.json', 'w') as outputFile:
    print('Writing: '+str(outputFile))
    json.dump(dataRestReviews, outputFile)
    outputFile.close()

#TOKENIZE AND POS

tweetTk = TweetTokenizer()
tk = nltk.data.load('tokenizers/punkt/english.pickle')


for line in dataRestReviews:
    newLine = copy.deepcopy(line)
    newLine['text'] = []
    for sentence in tk.tokenize(line['text']):
        x = nltk.pos_tag(tweetTk.tokenize(sentence))
        newLine['text'].append(x)
    dataReviewsPOS.append(newLine)

with open('Dataset/ReviewsOfResutrantsEdinburghPOS.json', 'w') as outputFile:
    print('Writing: '+str(outputFile))
    json.dump(dataReviewsPOS, outputFile)
    outputFile.close()











