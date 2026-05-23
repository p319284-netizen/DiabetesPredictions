#UNIVERSIDAD AUTONOMA DE CHIHUAHUA
#Facultad de Ingenieria

#MACHINE LEARNING END-TO-END PROJECT
#Dra. Olanda Prieto Ordaz

#Melissa Cirene Olivas Palma - 319284

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb
import sklearn
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path_modelo = os.path.join(BASE_DIR, "modelo_RandomForest.pkl")

def predictValue(some_data):
 #   with open('anaconda3/my_model1.pkl', 'rb') as f: 
    model = jb.load(path_modelo)
    data = model.predict(some_data)
    return data

st.text('UNIVERSIDAD AUTONOMA DE CHIHUAHUA')
st.text('Facultad de Ingenieria')
st.image('https://cimav.edu.mx/wp-content/uploads/2017/05/ingenieria-uach.png', width=100)

st.title('MACHINE LEARNING')
st.text('Dra. Olanda Prieto Ordaz')
st.text('Melissa Cirene Olivas Palma - 319284')
st.header('PREDICT DIABETES RISK PROBABILITY')

attrPregnancies = st.selectbox('Select Pregnancies', np.arange(0, 20, 1))
attrGlucose = st.selectbox('Select Glucose', np.arange(100, 250, 2))
attrBloodPressure = st.selectbox('Select BloodPressure', np.arange(60, 260, 1))
attrSkinThickness = st.selectbox('Select SkinThickness', np.arange(0, 100, 1))
attrInsulin = st.selectbox('Select Insulin', np.arange(1, 850, 1))
attrBMI = st.selectbox('Select BMI', np.arange(1, 70, 5))
attrDiabetesPedigree = st.selectbox('Select DiabetesPedigreeFunction', np.arange(0.0, 3.0, 0.1))
attrAge = st.selectbox('Select Age', np.arange(20, 85, 1))

if st.button('PREDICT VALUE'):
    data = pd.array([attrPregnancies, attrGlucose, attrBloodPressure, attrSkinThickness, attrInsulin, attrBMI, attrDiabetesPedigree, attrAge, 0])
    df = pd.DataFrame(data.reshape(1, 10), columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"])
    name1 =  predictValue(df)
    st.write(name1)
