from Cython.Compiler.PyrexTypes import best_match
from scipy.stats._continuous_distns import uniform_gen

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
    for review in dataEdinburghPOS:
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
    print(len(bestUsers))
    for i in range(len(bestUsers)):
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
            print('----------------------------------------------')
            print(rev['text'])
           # d+=rev['text']

    #print(d)
   # with open('output.txt','w') as out:
    #    out.write(d)



#printUserReviews('In6L6fy4jFlN0E-LEZXGiw','Edinburgh/ReviewsOfbusinessEdinburgh.json')
#printUserReviews('wx12_24dFiL1Pc0H_PygLw','Edinburgh/ReviewsOfbusinessEdinburgh.json')
#printUserReviews('7DxQDfrnoQI9nGALyi-LyQ','Edinburgh/ReviewsOfbusinessEdinburgh.json')

def calculateBigrams(dataEdinburghPOS,id, idValue):

    bigramsRank=[defaultdict(lambda: defaultdict(float)) for x in range (5)]

    #collect stop words to avoid ranking those
    cachedStopWords = stopwords.words("english")
    #ourWords=['"','lots','bar','bars','pub','pubs','restaurant','restaurants','place','places','bit','yum','food','meal','meals','friends','friend'
     #         ,'value','pm',':)','today','visit','thing','things','dishes','missus','favorite','way','point',
     #         'course','table','reason']
    ourWords=['"','\\',"'",'*','.','/','_',"isn't","that's",]
    for w in ourWords:
        cachedStopWords.append(w)
   #cachedStopWords.append('"').append("lots")

    numbOfReview = 0
    #for each review
    for review in  dataEdinburghPOS:
        #if the review is of that user
        if(review[id]==idValue):
            numbOfReview+=1
            bigramInReview=[defaultdict(float) for x in range (5)]
            bigCountPerReview = 0
            #for each sentence in the review
            for sentence in review['text']:
                #for each pair of word,tag in the sentence
                for idx, wordPlusTag in enumerate(sentence):
                #for wordPlusTag in sentence:
                    #split the word and tag
                    word = wordPlusTag[0]
                    tag = wordPlusTag[1]
                    #if the tag is a noun
                    #and the noun is not a stopword
                    if( (tag == 'NN' or tag == 'NNS')# or (tag=='RB' or tag=='RBR' or tag=='RBS') or (tag=='JJ' or tag=='JJS' or tag== 'JJR')# #or tag == 'JJ' or tag == 'JJS')
                       and not (word in cachedStopWords)):
                        rank=truncate(review['stars'])
                         #if there is a word after
                        if idx+1 < len(sentence) and (sentence[idx+1][1]=='JJ' or sentence[idx+1][1]=='JJS' or sentence[idx+1][1]== 'JJR') and not (sentence[idx+1][0] in cachedStopWords):

                            bigramsRank[rank-1][sentence[idx][0]+'_'+sentence[idx+1][0]]['count'] +=1
                            bigramInReview[rank-1][sentence[idx][0]+'_'+sentence[idx+1][0]] +=1
                            bigCountPerReview+=1
                        #if there is a word before
                        if idx-1 >= 0 and (sentence[idx-1][1]=='JJ' or sentence[idx-1][1]=='JJS' or sentence[idx-1][1]== 'JJR') and not (sentence[idx-1][0] in cachedStopWords):
                            bigramsRank[rank-1][sentence[idx][0]+'_'+sentence[idx-1][0]]['count'] +=1
                            bigramInReview[rank-1][sentence[idx][0]+'_'+sentence[idx-1][0]] +=1
                            bigCountPerReview+=1
            for i in range (5):
                for b in bigramInReview[i]:
                    bigramsRank[i][b]['regularity'] += bigramInReview[i][b]/bigCountPerReview

    for i in range(5):
        for b in bigramsRank[i]:
            bigramsRank[i][b]['regularity'] /=numbOfReview

    return bigramsRank

#a=calculateBigrams(dataEdinburghPOS,'user_id','7DxQDfrnoQI9nGALyi-LyQ')
#listOfCount=list(map(lambda x: [x[0],x[1]['regularity']], a[4].items()))
#countSorted=sorted(listOfCount,key=operator.itemgetter(1),reverse=True)




from math import log
def calculateUnigram(dataEdinburghPOS,id, idValue):

    totalWordsCount = [0 for x in range(5)]    # parole(NOMI!) usate in totale dall'utente per rank
    numberOfReviews = [0 for x in range(5)]    # review totali per un utente (ad edinburgh) per rank

    #create dictionary of user words
    #for each word there is a dictionary with its attributes
    unigramRank=[defaultdict(lambda: defaultdict(float)) for x in range (5)]

    #collect stop words to avoid ranking those
    cachedStopWords = stopwords.words("english")
   #cachedStopWords.append('"').append("lots")
    ourWords=['"','\\',"'",'*','.','/','_',"isn't","that's","it's","they're",'%',"we'd",":/","we'll"]

    # ourWords=['"','lots','bar','bars','pub','pubs','restaurant','restaurants','place','places','bit','yum','food','meal','meals','friends','friend'
    #           ,'value','pm',':)','today','visit','thing','things','dishes','missus','favorite','way','point',
    #           'course','table','reason']
    for w in ourWords:
        cachedStopWords.append(w)

    #for each review
    for review in dataEdinburghPOS:
        #if the review is of of that user
        if(review[id]==idValue):
            #reset the number of words in the current review
            countOfwordsInReview = 0
            #create dicotionary to store words of a single review
            wordsInReview = defaultdict(float)

            rank = truncate(review['stars'])-1

            #increase the number of reviews of that user
            numberOfReviews[rank] += 1
            #for each sentence in the review
            for sentence in review['text']:
                #for each pair of word,tag in the sentence
                for wordPlusTag in sentence:
                    #split the word and tag
                    word = wordPlusTag[0]
                    tag = wordPlusTag[1]
                    #if the tag is a noun
                    if( (tag == 'NN' or tag == 'NNS') #or (tag=='RB' or tag=='RBR' or tag=='RBS') or (tag=='JJ' or tag=='JJS' or tag== 'JJR')# #or tag == 'JJ' or tag == 'JJS')
                       and not (word in cachedStopWords)):
                        #increase the total number of words
                        totalWordsCount[rank] += 1
                        #increase the number of words in the review
                        countOfwordsInReview +=1
                        #increase the overall counter of that word in that rank
                        unigramRank[rank][word]['count'] +=1

                        #increase the count in the review
                        wordsInReview[word] += 1

            #update the regularity of words given this review
            for w in wordsInReview:
                unigramRank[rank][w]['regularity'] +=  wordsInReview[w]*(1/countOfwordsInReview)


    #normalizzo il numero di una parola sulle parole totali
    for i in range(5):
        for w in unigramRank[i]:
            unigramRank[i][w]['count'] /= totalWordsCount[i]
            unigramRank[i][w]['regularity'] /= numberOfReviews[i]
    return unigramRank



#b=calculateUnigram(dataEdinburghPOS,'user_id','7DxQDfrnoQI9nGALyi-LyQ')
#listOfCount=list(map(lambda x: [x[0],x[1]['count']], b[4].items()))
#listOfCount=list(map(lambda x: [x[0],log(x[1]['count'])+log(x[1]['regularity'])], b[4].items()))
#countSorted=sorted(listOfCount,key=operator.itemgetter(1),reverse=True)

def truncate(number):
    return int(str(number).split('.')[0])


def dfCreation2(data, folder, bestUsers, userFeatures, resBusiness):
    MULTIPLICATION_FACTOR= 100 #unigram count is multiplied so that we avoid underflow

    for i in range (len(bestUsers)):
        print(i)
        bestWordsOfUser=[] # list of best word of user for each rank

        for rank in range(5):
            bestWordsOfUser.append(userFeatures[bestUsers[i][0]][rank])  #HERE i can slice the number of words to consider
            #bestWordsOfUser.append(userFeatures['-8zhUSkiBdIRUfeXM1KM6Q'][rank])  #HERE i can slice the number of words to consider

        #====create the values for the user unflattening the all thing
        userValueList =  [val for sublist in bestWordsOfUser for val in sublist]
        userValueList = list(map(lambda x: list(x[1].values()),userValueList))
        userValueList =  [val*MULTIPLICATION_FACTOR for sublist in userValueList for val in sublist]

        df = pn.DataFrame()
        rankList =list()
        totalReview = 0
        for rev in data:
            if rev['user_id'] == bestUsers[i][0]:
            #if rev['user_id'] == '-8zhUSkiBdIRUfeXM1KM6Q':

                totalReview +=1
                businessCommon =[]
                for rank in range(5):
                    count =0
                    for word in  bestWordsOfUser[rank]: #number of words in that rank
                        #print(word[0],word[1])
                        #set(list(map(lambda x:x[0],bestWordsOfUser[rank]))) & set(list(resBusiness[rev['business_id']][rank].keys()))
                        if word[0] in list(resBusiness[rev['business_id']][rank].keys()):
                            businessCommon.append(list(resBusiness[rev['business_id']][rank][word[0]].values()))
                            #print(word[0],resBusiness[rev['business_id']][rank][word[0]])
                            count +=1
                        else:
                            businessCommon.append([0,0])

                    #print('rank:',rank,'total=',len(bestWordsOfUser[rank]),'common= ',count)

                tmp = [val*MULTIPLICATION_FACTOR for sublist in businessCommon for val in sublist]

                #=====unite user values with business values
                userPlusBusiness = userValueList + tmp
                df = df.append(pn.Series(userPlusBusiness), ignore_index=True)
                rankList.append(rev['stars'])
        #print('total review:',totalReview)
        #print('sparsity:',(df == 0).astype(int).sum(axis=1).sum(axis=0) / (len(df.columns)*len(df)))

        #save data file
        path = folder+'/' + bestUsers[i][0] + '.csv'
        df.to_csv(path, header=False, index_label=False, index=False)

        #save starts
        #======1 or 2
        starsPath =  folder+'/1-2/stars/' + bestUsers[i][0] + '.csv'
        tmpRankList=[]
        tmpRealRankList=[]
        for el in rankList:
            if el >=1 and el<=2:
                tmpRankList.append(el)
                tmpRealRankList.append(el)
            else:
                tmpRankList.append('-1') #other
                tmpRealRankList.append(el) #truth

        dfRank = pn.DataFrame(pn.Series(tmpRankList))
        dfRank = pn.concat([dfRank, pn.Series(tmpRealRankList)], axis=1)

        dfRank.to_csv(starsPath, header=False, index_label=False, index=False)

         #======3
        starsPath =  folder+'/3/stars/' + bestUsers[i][0] + '.csv'
        tmpRankList=[]
        tmpRealRankList=[]
        for el in rankList:
            if el >=3 and el <4:
                tmpRankList.append(el)
                tmpRealRankList.append(el)
            else:
                tmpRankList.append('-1') #other
                tmpRealRankList.append(el) #truth


        dfRank = pn.DataFrame(pn.Series(tmpRankList))
        dfRank = pn.concat([dfRank, pn.Series(tmpRealRankList)], axis=1)

        dfRank.to_csv(starsPath, header=False, index_label=False, index=False)

        #======4 or 5
        starsPath =  folder+'/4-5/stars/' + bestUsers[i][0] + '.csv'
        tmpRankList=[]
        tmpRealRankList=[]
        for el in rankList:
            if el >=4 and el<=5:
                tmpRankList.append(el)
                tmpRealRankList.append(el)
            else:
                tmpRankList.append('-1') #other
                tmpRealRankList.append(el) #truth

        dfRank = pn.DataFrame(pn.Series(tmpRankList))
        dfRank = pn.concat([dfRank, pn.Series(tmpRealRankList)], axis=1)

        dfRank.to_csv(starsPath, header=False, index_label=False, index=False)