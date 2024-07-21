import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib #for storing and loading models

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre']) #the input set
y = music_data['genre'] #the output data set

'''
model = DecisionTreeClassifier()
model.fit(X, y) #pass the entire data_set for the model
predictions = model.predict([ [21, 1], [22, 0] ]) #the prediction
predictions
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #split the data into a train and test set
model = DecisionTreeClassifier()
model.fit(X_train, y_train)                                            #provide the training data to the model
predictions = model.predict(X_test)                                    #the prediction the model makes
score = accuracy_score(y_test, predictions)                            #compares the accuracy of the predicted 
                                                                       #values and the actual values in the test data
score                                                                  #the changing values can be combated with cleaning the 
                                                                       #data and using a larger dataset

model = joblib.load('music-recommender.joblib')
predictions = model.predict([[21, 1]])
predictions