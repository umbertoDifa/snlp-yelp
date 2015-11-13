__author__ = 'Umberto'
from collections import defaultdict #has the extended version of dict
from nltk.corpus import stopwords

#return a dictionary of features for each noun used by the user in the reviews
#right now:
#noun
#noun frequency
#noun weighted frequency
#noun weighted rank
def calculateFeatures(dataEdinburghPOS,user_id):
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
        if(review['user_id']==user_id):
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
                    if( (tag=='NN' or tag == 'NNS' )
                       and (word not in cachedStopWords)):
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

