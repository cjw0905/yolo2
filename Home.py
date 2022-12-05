
import streamlit as st
from PIL import Image

st.set_page_config("TTok-Tak", "🎨")

st.sidebar.title('🎨 Main Menu')
st.title('🎨 Welcome To TTok-Tak')
st.subheader('당신의 취향을 반영한 디자인')

main_img = Image.open(str('C:/Users/최재원/Desktop/main.png'))
st.image(main_img, width = 700)
#
# st.subheader('소개글')
# st.write('소개글')

