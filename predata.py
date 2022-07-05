import pandas as pd
import numpy as np


def get_data():
    
    #create data frame from json file
    df1 = pd.read_json('Software.json', lines=True)
    df2 = pd.read_json('Prime_Pantry.json', lines=True)
    df = pd.concat([df1,df2],ignore_index=True)
    
    return df