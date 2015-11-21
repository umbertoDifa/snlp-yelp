__author__ = 'Umberto'
from collections import defaultdict #has the extended version of dict
from nltk.corpus import stopwords
import pandas as pn
import json
import csv
#return a dictionary of features for each noun used by the user in the reviews
#right now:
#noun
#noun frequency
#noun weighted frequency
#noun weighted rank
def calculateFeatures(dataEdinburghPOS,id, idValue):
    totalWordsCount = 0    # parole(NOMI!) usate in totale dall'utente
    numberOfReviews = 0    # review totali per un utente (ad edinburgh)

    #create dictionary of user words
    #for each word there is a dictionary with its attributes
    words = defaultdict(lambda: defaultdict(float))

    #collect stop words to avoid ranking those
    cachedStopWords = stopwords.words("english")
   #cachedStopWords.append('"').append("lots")

    #for each review
    for review in  dataEdinburghPOS:
        #if the review is of of that user
        if(review[id]==idValue):
            #reset the number of words in the current review
            countOfwordsInReview = 0
            #create dicotionary to store words of a single review
            wordsInReview = defaultdict(float)
            #increase the number of reviews of that user
            numberOfReviews += 1

            #for each sentence in the review
            for sentence in review['text']:
                #for each pair of word,tag in the sentence
                for wordPlusTag in sentence:
                    #split the word and tag
                    word = wordPlusTag[0]
                    tag = wordPlusTag[1]
                    #if the tag is a noun
                    if( (tag == 'NN' or tag == 'NNS') or (tag=='RB' or tag=='RBR' or tag=='RBS') or (tag=='JJ' or tag=='JJS' or tag== 'JJR')# #or tag == 'JJ' or tag == 'JJS')
                       and not (word in cachedStopWords)):
                        #increase the total number of words
                        totalWordsCount += 1
                        #increase the number of words in the review
                        countOfwordsInReview +=1
                        #increase the overall counter of that word
                        words[word]['count'] +=1
                        #increase the count in the review
                        wordsInReview[word] += 1

            #update the regularity of words given this review
            for w in wordsInReview:
                words[w]['regularity'] += wordsInReview[w]*(1/countOfwordsInReview)
                words[w]['relevance'] += wordsInReview[w]*(1/countOfwordsInReview) * (review['stars']/5)


    #normalizzo il numero di una parola sulle parole totali
    for w in words:
        words[w]['count'] = words[w]['count']/totalWordsCount
        words[w]['regularity'] /= numberOfReviews
        words[w]['relevance'] /= numberOfReviews
    return words



def dfCreation(data, folder, bestUsers, userFeatures, resBusiness):
    for i in range(len(bestUsers)):
        print(i)
        df = pn.DataFrame()
        dfRank = pn.DataFrame()
        for rev in data:
            if rev['user_id'] == bestUsers[i][0]:
                bestWordsOfUser = list(map(lambda x: x[0], userFeatures[bestUsers[i][0]]))
                common = list(filter(lambda x: x in bestWordsOfUser, resBusiness[rev['business_id']].keys()))
                # common = set(resBusiness[rev['business_id']]) & set(res[bestUsers[i][0]])
                userValuesDictionary = list(map(lambda x: x[1], userFeatures[bestUsers[i][0]]))

                tmp = list()
                for v in userValuesDictionary:
                    tmp.append(list(v.values()))

                for w in bestWordsOfUser:
                    if w in common:
                        tmp.append(list(resBusiness[rev['business_id']][w].values()))
                    else:
                        tmp.append([0, 0, 0])

                        # unflatten
                tmp = [val for sublist in tmp for val in sublist]
                # scale perche ci piace
                tmp = list(map(lambda x: x * 100, tmp))
                df = df.append(pn.Series(tmp), ignore_index=True)
                dfRank = dfRank.append(pn.Series(rev['stars']/5), ignore_index=True)
        path = folder+'/' + bestUsers[i][0] + '.csv'
        df.to_csv(path, header=False, index_label=False, index=False)
        pathRank = folder+'/stars/' + bestUsers[i][0] + '.csv'
        dfRank.to_csv(pathRank, header=False, index_label=False, index=False)


def printUserReviews(user,path):
    with open(path) as inputFile:
        data = json.load(inputFile)
    d=''
    for rev in data:
        #print(rev)
        if rev['user_id']==user:
          #  print('----------------------------------------------')
          #  print(rev['text'])
            d+=rev['text']

    print(d)
    with open('output.txt','w') as out:
        out.write(d)



#printUserReviews('In6L6fy4jFlN0E-LEZXGiw','Edinburgh/ReviewsOfbusinessEdinburgh.json')
printUserReviews('wx12_24dFiL1Pc0H_PygLw','Edinburgh/ReviewsOfbusinessEdinburgh.json')
printUserReviews('2pxcprc3GGAeI_RM88-Cgw','Edinburgh/ReviewsOfbusinessEdinburgh.json')