from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

def predict(X):
    
    #create label
    y = []
    for i in range(800):
        if i < 400:
            y.append(1)
        elif i >= 400:
            y.append(0)
            
    # print(y)
    # print(y[0], y[399], y[400], y[799])
    
    y = np.array(y).reshape(-1,1)
    
    # print(X.shape)
    # print(y.shape
    
    #split the data set into train data and test data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    model = svm.SVC()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    # print("test_label:predict_label")
    # print(f"{y_test}:{y_pred}")
    
    #evaluate model
    print(f"accuracy :{accuracy_score(y_test, y_pred)}")
    print(f"precision:{precision_score(y_test, y_pred)}")
    print(f"recall   :{recall_score(y_test, y_pred)}")