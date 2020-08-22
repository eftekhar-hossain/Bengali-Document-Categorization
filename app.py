from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import re, pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	stopword_list = open('bangla_stopwords.pkl', 'rb')
	stp = pickle.load(stopword_list)

	def process_news(articles):
		news = articles.replace('\n', ' ')
		news = re.sub('[^\u0980-\u09FF]', ' ', str(news))  # removing unnecessary punctuation
		# stopwords removal
		result = news.split()
		news = [word.strip() for word in result if word not in stp]
		news = " ".join(news)
		return news

	# load the saved tokenizer
	with open('tokenizer.pickle', 'rb') as handle:
		loaded_tokenizer = pickle.load(handle)

	# load the CNN-BiLSTM model
	model = tf.keras.models.load_model('Document_Categorization.h5')

	# List of news categories
	class_names = ['Accident', 'Art', 'Crime', 'Economics', 'Education', 'Entertainment',
				   'Environment', 'International', 'Opinion', 'Politics', 'Science', 'Sports']

	if request.method == 'POST':
		article = request.form['news']
		cleaned_news = process_news(article)
		seq = loaded_tokenizer.texts_to_sequences([cleaned_news])
		padded = pad_sequences(seq, value=0.0, padding='post',maxlen=300)
		pred = model.predict(padded)
		category_name = class_names[np.argmax(pred)]

		#score = round(max(prediction_score.reshape(-1)), 2) * 100

	return render_template('predict.html', category=category_name)



if __name__ == '__main__':
	app.run(debug=True)