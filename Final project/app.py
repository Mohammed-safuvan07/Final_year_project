import streamlit as st
import numpy as np
import pandas as pd
import pickle
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import os
from matplotlib import image
from sentence_transformers import SentenceTransformer, util

st.write("<h1 style='color:tomato;text-align:center;'>Fake News Detection Model</h1>",unsafe_allow_html=True)


st.snow()



tf_model = pickle.load(open('C:/Users/Safuvan/bertfile.sav','rb'))
load_model = pickle.load(open('C:/Users/Safuvan/bert.sav','rb'))
#data = pd.read_csv(DATA_PATH)

#img = image.imread(image_path)
#st.image(img)

st.write("<h3 style='color:;'>Enter a News for Prediction</h3>",unsafe_allow_html=True)
news_input = st.text_area('')

dat ={'news':[news_input]}
data2 = pd.DataFrame(dat)
new1 = data2[['news']]

def preprocess(pro):
    process = re.sub('[^a-zA-Z]'," ",pro)
    lowe = process.lower()
    tokens = lowe.split()
   
    stop = [lemmatizer.lemmatize(i) for i in tokens if i not in stopwords.words('English')]
    lemmas =pd.Series([ " ".join(stop)])
    return lemmas

pnews = new1['news'].apply(preprocess)
pnews.columns=['news']   

newtf1 = tf_model.encode(pnews['news']) 

submit = st.button('predict')
load_model.predict(newtf1)
#submit == True
#st.success(tf_model.transform(pnews['news']))

if (submit==True and len(news_input)== 0):
    st.write('Enter  a valid news')

    st.balloons()
elif  (submit==True and len(news_input)<=250):
      st.write('It is not  a valid news')    
else:
      prediction=(load_model.predict(newtf1))
      if prediction == 1:
        st.write('news is True')
      else:
           st.write('news is False')  

#else :
 #       st.write(':red[This news is Fake]')
#else:
#    st.write('It is not a valid news')

#st.subheader(':orange[News Dataset]')

#with st.sidebar:
#    st.dataframe(data)    
        
    
        


