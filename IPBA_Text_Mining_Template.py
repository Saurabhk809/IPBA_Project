# How to build a count vectoriser
# A count vectorises is a numerical form of a text document

#set the display options
import pandas as pd
pd.set_option('display.max_columns', 1000, 'display.width', 1000, 'display.max_rows',1000)
pd.set_option('display.float_format',lambda x:'%.10f' %x)

from sklearn.feature_extraction.text import CountVectorizer

def featurenames(cv,matrix):
    # don't do this on bigger matrix
    import pandas as pd
    output=pd.DataFrame(matrix,columns=cv.get_feature_names_out())
    print('\n',output)

def createcvmatrix(corpus,i,j):
    #cv=CountVectorizer(ngram_range=(i,j),stop_words='english')
    cv = CountVectorizer(ngram_range=(i, j))
    matrix=cv.fit_transform(corpus).toarray()
    featurenames(cv,matrix)

def tfidf_vectoriser(corpus,i,j):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(i,j))
    tfidf_features = tfidf_vectorizer.fit_transform(corpus)

    # Get the feature names (words)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Get the TF-IDF scores for each word
    tfidf_scores = tfidf_features.toarray()

    # Print the words and their corresponding TF-IDF scores
    for word, score in zip(feature_names, tfidf_scores):
        print('tfidf score :\n',{word:score})

def main():
    corpus = ['this is sentence one','this is sentence two','this is sentence three']

    # Using CV matrix
    createcvmatrix(corpus,1,1) # 1 work gram & 1 word bigram matrix
    createcvmatrix(corpus,1,2) # 1 word gram & 2 word bigram
    #createcvmatrix(corpus,2,2) # 2 word gram & 2 word Bigram matrix
    createcvmatrix(corpus,1,3) # 1 word gram & 3 word Trigram matrix

    # Using tfidx vectoriser
    tfidf_vectoriser(corpus,1,1)
    tfidf_vectoriser(corpus,1,2)
    tfidf_vectoriser(corpus,1,3)

main()