import pandas as pd
import numpy as np

from predata import create_tfidf_table
from predict import predict



if __name__ == "__main__":
    
    input_documents = create_tfidf_table()
    
    predict(input_documents)