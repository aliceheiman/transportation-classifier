from fastcore.all import *
from fastai.vision.all import *
import streamlit as st

## LOAD MODEl
learn_inf = load_learner("export.pkl")

## CLASSIFIER
def classify_img(data):
    pred, pred_idx, probs = learn_inf.predict(data)
    return pred, probs[pred_idx]


## STREAMLIT
st.title("Transportation Classifier! 🚴‍♀️🛹🛼")
st.markdown("This is a machine learning model capable of classifying images of different modes of transportation, for example bikes, rollerskates, trains,  and boats. The model leverages transfer learning and was fine tuned on the Resnet34 neural network. Feel free to experiment!")

bytes_data = None


uploaded_image = st.file_uploader("Choose an image of a mode of transportation (e.g. bike, rollerskates, bus):")
if uploaded_image:
    bytes_data = uploaded_image.getvalue()

    st.image(bytes_data, caption="Uploaded image")   

if bytes_data:
    classify = st.button("CLASSIFY!")
    if classify:
        label, confidence = classify_img(bytes_data)

        st.write(f"It is a {label}! ({confidence:.04f})")