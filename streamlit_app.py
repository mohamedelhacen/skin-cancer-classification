import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

classes = ['Basal Cell Carcinoma (BCC)', 'Squamous Cell Carcinoma (SCC)', 
           'Actinic Keratosis (ACK)', 'Seborrheic Keratosis (SEK)', 'Melanoma (MEL)', 
           'Nevus (NEV)']
s = ''
for i in classes:
    s += "- " + i + "\n"

st.title("Skin Cancer Classification")
st.write(f"This model can classify the skin images to one of this deseases bellow:")
st.markdown(s)

st.write("## Upload a skin image")
uploaded_image = st.file_uploader("Upload a skin image and press *Classify* button", type=['jpg', 'jpeg', 'png'])

if uploaded_image:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(uploaded_image)

    with col2:
        if st.button('Classify'):
            
            orginal_image = Image.open(uploaded_image).convert('RGB')
            image = np.array(orginal_image.copy())
            image = cv2.resize(image, (512, 512))

            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize(128)])
            image = transform(image)
            image = image.unsqueeze(0)

            model = torch.load('model.pth')
            class_to_idx = model.class_to_idx
            with torch.no_grad():
                outputs = model(image.float())
            _, pred = torch.max(outputs, 1)
            
            st.write(f'This image has been classified to **{list(class_to_idx.keys())[pred.item()]}** skin lesion.')


st.write("# About the data ...")
st.markdown("The dataset used here is [PAD-UFES-20](https://www.kaggle.com/datasets/mahdavi1202/skin-cancer)")
st.markdown('### Samples')

col3, col4, col5 = st.columns([1, 1, 1])
with col3:
    st.image('samples/PAT_1042_187_746.png')
with col4:
    st.image('samples/PAT_1105_420_520.png')
with col5:
    st.image('samples/PAT_1129_498_930.png')
col6, col7, col8 = st.columns([1, 1, 1])
with col6:
    st.image('samples/PAT_1148_542_708.png')
with col7:
    st.image('samples/PAT_1130_501_181.png')
with col8:
    st.image('samples/PAT_1107_427_352.png')

st.write('### Images distribution per class')
st.image('description.png')

st.write("# About the network ...")
st.write('I have used here a fine-tuned **ResNet18** netowrk')
st.write("### Run the code")
st.write("First downlowd the data from [Kaggle](https://www.kaggle.com/datasets/mahdavi1202/skin-cancer)")
st.write("Then clone the code from [Github](https://github.com/mohamedelhacen/skin-cancer-classification)")
st.write('Run the following command to prepare the data, train and test the network, and save checkpoints')
st.write('`pyton main.py --train.epochs 100`')