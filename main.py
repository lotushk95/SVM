import pandas as pd
import numpy as np

from predata import get_data
from predict import predict
from tfidf import calc_tfidf



if __name__ == "__main__":
    
    df = get_data()
    
    tfidf_table = calc_tfidf(df)
    
    predict(tfidf_table)