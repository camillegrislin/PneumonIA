import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2

st.set_option('deprecation.showfileUploaderEncoding', False)

with st.sidebar:
        st.write("Hello our names are Zoé DUPRAT, Martin CORNEN, Camille GRISLIN.") 
        st.write("We decided to create an app where you can drag your chest x-ray and an IA will determine your pourcentage of chance to have a pneumonia.")

def predict(testing_image):
    
    model = load_model('model_ML.h5')
 
    image = Image.open(testing_image).convert('RGB')
    image = image.resize((224,224))
    image = img_to_array(image)
    image = image.reshape(1,224,224,3)
    image = image/255.0

    result = model.predict(image)

    result = np.argmax(result, axis=1)
    

    if result == 0:
        return "Patient is Normal."
    else :
       return "Patient has pneumonia."
   
    

def main():
    st.title('Covid-Pneumonia Detection')
    st.subheader('This project will predict whether a person is suffering from Covid or Viral Pneumonia using Radiograph images.')

    image = st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])

    if image is not None :

        #to view uploaded image
        st.image(Image.open(image))

        # Prediction
        if st.button('Result', help='Prediction'):
            st.success(predict(image))

if __name__=='__main__':
    main()
