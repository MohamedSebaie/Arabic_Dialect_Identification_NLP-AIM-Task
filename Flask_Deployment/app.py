from flask import Flask, render_template, request
import joblib
import re
import nltk
import numpy as np
import pandas as pd
import aranorm
import preprocess_arabert

# __name__ is equal to app.py
app = Flask(__name__)

# load model from model.pck
model = joblib.load('LinearSVC_model.pkl')



@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
	keys_dictionary   = {0:"Egypt",1:"Palestine",2:"Kuwait",3:"Libya",4:"Qatar",5:"Jordan",6:"Lebanon",7:"Saudi Arabia",
						8:"United Arab Emirates",9:"Bahrain",10:"Oman",11:"Syria",12:"Algeria",13:"Iraq",14:"Sudan",15:"Morocco",
						16:"Yemen",17:"Tunisia"}

	text =  [request.form['tweet']]
	df = pd.DataFrame({'T':text})

	def tweet_preprcessing(tweet):
	    text_preprocessed = preprocess_arabert.preprocess(tweet, do_farasa_tokenization=True)
	    preprocessed_tweet= aranorm.normalize_arabic_text(text_preprocessed)
	    return preprocessed_tweet

	df['T'] = df['T'].apply(tweet_preprcessing)

	dialect = keys_dictionary.get(model.predict(df['T'])[0])
	
	return render_template("index.html", dialect=dialect)	


if __name__ == "__main__":
    app.run()
