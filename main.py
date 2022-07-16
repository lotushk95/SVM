import pandas as pd
import numpy as np

from predata import get_data
from predict import predict
from tfidf import calc_tfidf

if __name__ == "__main__":
    
    #read data from json file
    df = get_data()
    
    #calculate tfidf
    tfidf_table = calc_tfidf(df)
    
    #classify documents
    predict(tfidf_table)