import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas

def calc_tfidf(df):
    review_text = []
    for data in df['reviewText'].values.astype('U'):
        review_text.append(data)
        
    filtered_text = []
    for text in review_text:
        
        word_list = []
        token = nltk.word_tokenize(text)
        token_tag = nltk.pos_tag(token)
        # print(token_tag)
        
        for word, tag in token_tag:
            if tag == 'JJ' or tag == 'JJS' or tag == 'JJR' or tag == 'RB' or tag == 'NN' or tag == 'NNP' or tag == 'NNS' or tag == 'VB' or tag == 'VBD' or tag == 'VBG' or tag == 'VBN' or tag == 'VBZ':
                word_list.append(word)
            else:
                continue
        filtered_text.append(' '.join(word_list))

    

    # calculate TF-IDF and vectorize
    vectorizer = TfidfVectorizer()

    # X = vectorizer.fit_transform(df['reviewText'].values.astype('U'))
    X = vectorizer.fit_transform(filtered_text)
    
    input_documents = X.toarray()
    
    return input_documents