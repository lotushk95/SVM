from collections import Counter
from collections import defaultdict
import math
import numpy as np

def calc_tfidf(input_documents):
    N = len(input_documents)
    words = "".join(input_documents).split()
    count = Counter(words).most_common()
    
    #build dictionaries
    rdic = [i[0] for i in count]
    
    #calculating TFIDF
    document_TFtable = defaultdict(Counter)
    document_DFtable = Counter()
    document_TFIDFtable = defaultdict(Counter)
    
    #calculate term frequency
    for document in input_documents:
        words = document.split()
        for word in words:
            document_TFtable[document][word] += 1
            
    #calculate document frequency
        for kw in document_TFtable[document].keys():
            document_DFtable[kw] += 1
            
    #calculate TF-IDF
    for document in input_documents:
        for kw in document_TFtable[document].keys():
            document_TFIDFtable[document][kw] = document_TFtable[document][kw] * math.log(N/document_DFtable[kw])
            
          
    # make vector for calculating cosine similarity
    TFIDFtable = [[0.0 for _ in range(len(rdic))] for _ in range(len(document_TFIDFtable))]

    for i in range(len(input_documents)):
        for j in range(len(rdic)):
            keys_array = document_TFIDFtable[input_documents[i]].keys()
            for key in keys_array:
                if rdic[j] == key:
                    TFIDFtable[i][j] = document_TFIDFtable[input_documents[i]][key]
    
    return TFIDFtable #return vector for cosine similarity