#UNIVERSIDAD AUTONOMA DE CHIHUAHUA
#Facultad de Ingenieria

#MACHINE LEARNING END-TO-END PROJECT
#Dra. Olanda Prieto Ordaz

#Melissa Cirene Olivas Palma - 319284

import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb
import sklearn
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin

col_names = "total_rooms", "total_bedrooms", "population", "households"
rooms_ix, bedrooms_ix, population_ix, households_ix = [1,2,3,4]

#def predictValue(some_data):
 #   with open('anaconda3/my_model1.pkl', 'rb') as f: 
  #      model = jb.load(f)
   #     data = model.predict(some_data)
   # return data

st.text('UNIVERSIDAD AUTONOMA DE CHIHUAHUA')
st.text('Facultad de Ingenieria')
st.image('https://cimav.edu.mx/wp-content/uploads/2017/05/ingenieria-uach.png', width=100)

st.title('MACHINE LEARNING')
st.text('Dra. Olanda Prieto Ordaz')
st.text('Melissa Cirene Olivas Palma - 319284')
st.header('PREDICT DIABETES RISK PROBABILITY')

attrPregnancies = st.selectbox('Select Pregnancies', np.arange(-130, -110, 2))
attrGlucose = st.selectbox('Select Glucose', np.arange(30, 50, 2))
attrBloodPressure = st.selectbox('Select BloodPressure', np.arange(1, 60, 1))
attrSkinThickness = st.selectbox('Select SkinThickness', np.arange(2, 39320, 1))
attrInsulin = st.selectbox('Select Insulin', np.arange(1, 6445, 1))
attrBMI = st.selectbox('Select BMI', np.arange(1, 40000, 5))
attrDiabetesPedigree = st.selectbox('Select DiabetesPedigreeFunction', np.arange(1, 6082, 1))
attrAge = st.selectbox('Select Age', np.arange(0.0, 15.00, 0.1))

#if st.button('PREDICT VALUE'):
 #   data = pd.array([attrPregnancies, attrGlucose, attrBloodPressure, attrSkinThickness, attrInsulin, attrBMI, attrDiabetesPedigree, attrAge, 0])
  #  df = pd.DataFrame(data.reshape(1, 10), columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"])
   # name1 =  predictValue(df)
    #st.write(name1)
