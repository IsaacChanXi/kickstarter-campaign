from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	nltk.download('stopwords')
	NB_spam_model = open('model.pkl','rb')
	clf = joblib.load(NB_spam_model)

	if request.method == 'POST':
                pickle_in = open("cv.pickle","rb")
                cv = pickle.load(pickle_in)
                text = request.form['message']
                review = text.split()
                ps = PorterStemmer()
                review= [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
                message=" ".join(review)
                data = [message]
                vect = cv.transform(data).toarray()
                my_prediction = str(round(clf.predict_proba(vect)[0][1]*100,1))+'%'
	return render_template('result.html', prediction = my_prediction)

if __name__ == '__main__':
        app.run(debug=True)
