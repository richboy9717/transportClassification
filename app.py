import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title("Transport Classifier")

file = st.file_uploader("Rasm yuklash", type=["jpg", "png"])

if file:
    st.image(file)

    img = PILImage.create(file)

    model = load_learner("./transport_model.pkl")

    # predict

    pred, pred_id, probs = model.predict(img)


    st.success(f'Bashorat: {pred}')
    st.info(f'Extimolligi: {probs[pred_id]}')

    # plot
    fig = px.bar(x=probs, y=model.dls.vocab, orientation='h')
    fig.update_layout(title="Extimolliklar")
    st.plotly_chart(fig)
