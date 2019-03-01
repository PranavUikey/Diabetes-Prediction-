from flask import Flask,request,json
import pandas as pd
from keras import models
from keras import layers
#from keras import layers.D
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
app = Flask(__name__)

@app.route('/',methods =['GET','POST'])

def index():
	df = pd.read_csv('diabetes.csv')
	#df.head()
	df_labels = df['Outcome']
	df_features = df.drop('Outcome',1)
	df_features.replace('?', -99999, inplace=True)
	#for i in df_features.columns:
    	#print(i)
    	#print(df_features[i].nunique)
	label = []
	for i in df_labels:
		if(i ==1):
			label.append([1,0])
		elif(i==0):
			label.append([0,1])
	df_features = np.array(df_features)
	label =np.array(label)
	from sklearn.model_selection import train_test_split
	X_train,X_test,y_train,y_test = train_test_split(df_features,label,random_state=42,test_size = 0.3)
	scaler = MinMaxScaler(feature_range=(0,1))
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)
	"""from numpy.random import seed
				seed(1)
				model = models.Sequential()
				model.add(layers.Dense(500, activation='sigmoid', input_dim=8))
				model.add(layers.Dense(100, activation='sigmoid'))
				model.add(layers.Dense(2, activation='softmax'))
				model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
				model.fit(X_train_scaled,y_train,epochs=1000,batch_size =10)
				model.save('diabetes.h5')"""
	#model
	model = models.load_model('daibetes.h5')
	result = model.predict_classes(np.array([X_test[1]]))
	if result == 0 :
		a = 'No Diabetes'
	else:
		a = 'Diabetes'
	return json.dumps(a)
	


if __name__ =="__main__":
	app.run(debug = True)