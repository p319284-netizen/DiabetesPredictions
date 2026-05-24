#UNIVERSIDAD AUTONOMA DE CHIHUAHUA
#Facultad de Ingenieria

#MACHINE LEARNING END-TO-END PROJECT
#Dra. Olanda Prieto Ordaz

#Melissa Cirene Olivas Palma - 319284

# =====================================================================
# IMPORTACIÓN DE LIBRERIAS UTILIZADAS
# =====================================================================
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
import joblib as jb
from PIL import Image
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats.mstats import winsorize
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin,  ClassifierMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

# =====================================================================
# FUNCIONES PERSONALIZADAS DEL PREPROCESAMIENTO
# =====================================================================

columns_for_scaling = ['Pregnancies', 'Age', 'DiabetesPedigreeFunction']
columns_with_null_values = ['Insulin', 'Glucose', 'BMI']
columns_for_mix = ['BloodPressure', 'SkinThickness']

class mixerColumn(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    glucose = X[:, 0]
    insulin = X[:, 1]

    # Calculate the new feature.
    glucose_insulin = glucose * insulin

    # Return combined features (Glucose / Insulin)
    return np.hstack(glucose_insulin.reshape(1, -1, 1))
   
def aplicar_winsorize(X, limite_superior=0.05):
    # Ensure X is 2D, even if it's a single column
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    X_winsorized = np.empty_like(X, dtype=float)
    for i in range(X.shape[1]):
        X_winsorized[:, i] = winsorize(X[:, i], limits=(None, limite_superior))
    return X_winsorized

def replace_zeros_with_nan(X):
    if X.ndim == 1:
        return np.where(X == 0, np.nan, X).reshape(-1, 1)
    return np.where(X == 0, np.nan, X)

pipeline_standard_numeric = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('quantile', FunctionTransformer(func=aplicar_winsorize, kw_args={'limite_superior': 0.10}, validate=False)),
    ('scaler', StandardScaler()) ])

pipeline_with_null_processing = Pipeline(steps=[
    ('replace_empty', FunctionTransformer(func=replace_zeros_with_nan, validate=False)),
    ('imputer', SimpleImputer(strategy='median')),
    ('quantile', FunctionTransformer(func=aplicar_winsorize, kw_args={'limite_superior': 0.10}, validate=False)),
    ('scaler', StandardScaler()) ])

pipeline_for_mix = Pipeline(steps=[
    ('replace_empty', FunctionTransformer(func=replace_zeros_with_nan, validate=False)),
    ('imputer', SimpleImputer(strategy='median')),
    ('quantile', FunctionTransformer(func=aplicar_winsorize, kw_args={'limite_superior': 0.05}, validate=False)),
    ('mixer_column', mixerColumn()),
    ('scaler', StandardScaler()) ])

preprocesador = ColumnTransformer(
    transformers=[
        ('standard_num_proc', pipeline_standard_numeric, columns_for_scaling),
        ('total_charges_proc', pipeline_with_null_processing, columns_with_null_values),
        ('mixer_columns', pipeline_for_mix, columns_for_mix),
    ],    remainder='drop')

# =====================================================================
# IMPORTACION DE LOS MODELOS
# =====================================================================

modelRandomForest = jb.load("modelo_RandomForest.pkl")
modelAdaBoost = jb.load("modelo_AdaBoost.pkl")
modelXGBoost = jb.load("modelo_GXBoost.pkl")

def predictValue(some_data):
    predRandomForest = modelRandomForest.predict(some_data)
    predAdaBoost = modelAdaBoost.predict(some_data)
    predXGBoost = modelXGBoost.predict(some_data)
    return predRandomForest, predAdaBoost, predXGBoost


st.sidebar.title("UNIVERSIDAD AUTONOMA DE CHIHUAHUA")
st.sidebar.caption("Facultad de Ingeniería")
st.sidebar.image("https://cimav.edu.mx/wp-content/uploads/2017/05/ingenieria-uach.png', width=50")

#st.text('UNIVERSIDAD AUTONOMA DE CHIHUAHUA')
#st.text('Facultad de Ingenieria')
#st.image('https://cimav.edu.mx/wp-content/uploads/2017/05/ingenieria-uach.png', width=100)

st.title('MACHINE LEARNING')
st.text('Dra. Olanda Prieto Ordaz')
st.text('Melissa Cirene Olivas Palma - 319284')
st.header('PREDICT DIABETES RISK PROBABILITY')

col1, col2 = st.columns(2)
with col1:
    attrPregnancies = st.selectbox('Select Pregnancies', np.arange(0, 20, 1))

with col2:
    attrGlucose = st.selectbox('Select Glucose', np.arange(100, 250, 2))

col3, col4 = st.columns(2)
with col3:
    attrBloodPressure = st.selectbox('Select BloodPressure', np.arange(60, 260, 1))

with col4:
    attrSkinThickness = st.selectbox('Select SkinThickness', np.arange(0, 100, 1))

col5, col6 = st.columns(2)
with col5:
    attrInsulin = st.selectbox('Select Insulin', np.arange(1, 850, 1))

with col6:
    attrBMI = st.selectbox('Select BMI', np.arange(1, 70, 0.1))

col7, col8 = st.columns(2)
with col5:
    attrDiabetesPedigree = st.selectbox('Select DiabetesPedigreeFunction', np.arange(0.0, 3.0, 0.1))

with col6:
    attrAge = st.selectbox('Select Age', np.arange(20, 85, 1))


# =====================================================================
# PREDICCIONES
# =====================================================================

if st.button('PREDICT VALUE'):
    data = pd.array([attrPregnancies, attrGlucose, attrBloodPressure, attrSkinThickness, attrInsulin, attrBMI, attrDiabetesPedigree, attrAge])
    df = pd.DataFrame(data.reshape(1, 8), columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])
    name1, name2, name3 =  predictValue(df)
    
    col10, col11, col12 = st.columns(3)
    with col10:
        st.metric("Random Forest", f"{name1}" )
    with col11:
        st.metric("AdaBoost", f"{name2}")
    with col12:
        st.metric("XGBoost", f"{name3}")
