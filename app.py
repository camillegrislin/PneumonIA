import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2

st.set_option('deprecation.showfileUploaderEncoding', False)

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
   
## Sidebar

st.sidebar.image("pnemonIApp_logo.png")
st.sidebar.markdown(
         " ## PneumonIApp")
st.sidebar.markdown(
         "created by Martin CORNEN, Zoé DUPRAT, Camille Grislin")
st.sidebar.markdown(
         "click *on [Github](https://github.com/camillegrislin/PneumonIA)* to see the code")

def main():
    st.title('Pneumonia Detection')
    st.subheader('This project will predict whether a person is suffering from Pneumonia using Radiograph images.')

    image = st.file_uploader('Upload your radiograph images', type=['jpg', 'jpeg', 'png'])

    if image is not None :

        #to view uploaded image
        st.image(Image.open(image))

        # Prediction
        if st.button('Result', help='Prediction'):
            st.success(predict(image))

if __name__=='__main__':
    main()
