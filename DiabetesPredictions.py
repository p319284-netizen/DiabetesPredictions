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

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def predictValue(some_data):
    with open('anaconda3/my_model1.pkl', 'rb') as f: 
        model = jb.load(f)
        data = model.predict(some_data)
    return data

st.text('UNIVERSIDAD AUTONOMA DE CHIHUAHUA')
st.text('Facultad de Ingenieria')
st.image('https://cimav.edu.mx/wp-content/uploads/2017/05/ingenieria-uach.png', width=100)

st.title('MACHINE LEARNING')
st.text('Dra. Olanda Prieto Ordaz')
st.text('Melissa Cirene Olivas Palma - 319284')
st.header('MEDIAN HOUSE VALUE IN CAROLINA')

dtAttributes = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']

attrLongitude = st.selectbox('Select Longitude', np.arange(-130, -110, 2))
attrLatitude = st.selectbox('Select Latitude', np.arange(30, 50, 2))
attrMedianAge = st.selectbox('Select Median Age', np.arange(1, 60, 1))
attrTotalRooms = st.selectbox('Select Total Rooms', np.arange(2, 39320, 1))
attrTotalBedrooms = st.selectbox('Select Total Bedrooms', np.arange(1, 6445, 1))
attrPopulation = st.selectbox('Select Population', np.arange(1, 40000, 5))
attrHouseholds = st.selectbox('Select Households', np.arange(1, 6082, 1))
attrMediamIncome = st.selectbox('Select Median Income', np.arange(0.0, 15.00, 0.1))
attrOceanProx = st.selectbox('Select Ocean Proximity', dtAttributes)

#if st.button('PREDICT VALUE'):
 #   data = pd.array([attrLongitude, attrLatitude, attrMedianAge, attrTotalRooms, attrTotalBedrooms, attrPopulation, attrHouseholds, attrMediamIncome, attrOceanProx, 0])
  #  df = pd.DataFrame(data.reshape(1, 10), columns = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity", "median_house_value"])
   # name1 =  predictValue(df)
    #st.write(name1)