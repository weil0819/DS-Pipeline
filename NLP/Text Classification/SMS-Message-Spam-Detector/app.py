#!/usr/bin/python
# -*- coding: utf-8 -*- 

"""
The app.py file contains the main code that will be executed 
by the Python interpreter to run the Flask web application, 
it included the ML code for classifying SMS messages

@date: Fri 6 Mar. 2020
"""

# Import necessary modules.
import pickle
from flask import Flask,render_template,url_for,request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib 


app = Flask(__name__)		

# Specify the URL that should trigger the execution of the 'home' function.
@app.route('/')
def home():
	# Rendered the home.html HTML file.
	return render_template('home.html')


# Use the POST method to transport the form data to the server in the message body.
@app.route('/predict',methods=['POST'])	# Only accept POST method to transfer data
def predict():
	# Extract Feature With CountVectorizer.
	cv = CountVectorizer()

	# Usage of Saved Model.
	NB_spam_model = open('NB_spam_model.pkl','rb')
	clf = joblib.load(NB_spam_model)

	if request.method == 'POST':
		message = request.form['message']	# parser
		data = [message]
		vect = cv.transform(data).toarray()	# testing vector data
		my_prediction = clf.predict(vect)	# predicting
	return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)




