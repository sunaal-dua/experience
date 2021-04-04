from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

def data_vectorizer(xtr,xte,maxdf=0.60,max_feat=1000,vectorizer="bow"):
  '''
  Converts the train and test set into BagOfWords or Tf-Idf matrix. Returns two arrays for train set and test set
  of size (document size,max_feat)
  
  xtr: train set (array)

  xte: test set (array)

  maxdf: ignores terms/words that have a document frequency higher than 'maxdf'. In other words, it ignores terms that
  appear in more than 'maxdf' number of documents/records. Give value between 0 and 1. Default value = 0.6(60%)

  max_feat: builds a vocabulary considering only top "max_features" ordered by term frequency across the corpus. Default value = 1000

  vectorizer: enter 'bow' for bag of words transformation or 'tfidf' for Tf-Idf transformation. Default value = 'bow' (string)
  '''

  if(vectorizer=="bow"):
    vec = CountVectorizer(max_df=maxdf, max_features=max_feat)
  elif(vectorizer=="tfidf"):
    vec = TfidfVectorizer(max_df=maxdf, max_features=max_feat)
  else:
    print("Enter 'bow' for bag of words transformation or 'tfidf' for Tf-Idf transformation in vectorizer parameter")
    err = np.array(["unknown","unknown"])
    return err

  xtr_vec = vec.fit_transform(xtr).toarray()
  xte_vec = vec.transform(xte).toarray()
  return xtr_vec, xte_vec