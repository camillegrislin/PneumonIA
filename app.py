import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import time

st.set_option('deprecation.showfileUploaderEncoding', False)

def predict(testing_image):
    
    model = load_model('model_ML_F.h5')
 
    image = Image.open(testing_image).convert('RGB')
    image = image.resize((224,224))
    image = img_to_array(image)
    image = image.reshape(1,224,224,3)
    image = image/255.0
    
    result = model.predict(image)
    result = np.argmax(result, axis=1)
    
    if result == 0:
       return st.success("Patient is Normal.")
    else :
       return st.error("Patient has pneumonia.")
   

# Layout

st.title('Do you have pneumonia ?')
st.subheader('Give us a radiograph and we will predict whether you are suffering from Pneumonia or not.')
image = st.file_uploader('Upload your radiograph images here', type=['jpg', 'jpeg', 'png'])
if image is not None :

        st.image(Image.open(image))
        st.markdown("See the result by clicking bellow")
        if st.button('Here', help='Prediction'):
            with st.spinner(text="Work in Progress"):
                (predict(image))
       
            
        

## Sidebar

st.sidebar.image("logo_pneumoIApp.png")
st.sidebar.markdown("# PneumonIApp")
st.sidebar.markdown("created by Martin CORNEN, Zoé DUPRAT, Camille Grislin")
st.sidebar.markdown("click *on [Github](https://github.com/camillegrislin/PneumonIA)* to see the code")
st.sidebar.markdown("---")
st.sidebar.markdown("### About the projet")
st.sidebar.markdown("This project was made for the DataCamp course. The goal was to create a web app using Streamlit and deploy it on Streamlit Cloud. It detects pneumonia using radiograph images thanks to AI and particulary neural networks. We used a preconstruct model named VGG19 and trained it on our data.")
st.sidebar.markdown("---")

