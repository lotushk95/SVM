import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_table():
    #create data frame from json file
    df1 = pd.read_json('Software.json', lines=True)
    df2 = pd.read_json('Prime_Pantry.json', lines=True)
    
    df = pd.concat([df1,df2],ignore_index=True)
    
    # calculate TF-IDF and vectorize
    vectorizer = TfidfVectorizer()
    
    X = vectorizer.fit_transform(df['reviewText'].values.astype('U'))
    
    input_documents = X.toarray()
    
    return input_documents