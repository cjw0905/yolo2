
import streamlit as st
from PIL import Image

st.set_page_config("TTok-Tak", "π¨")

st.sidebar.title('π¨ Main Menu')
st.title('π¨ Welcome To TTok-Tak')
st.subheader('λΉμ μ μ·¨ν₯μ λ°μν λμμΈ')

main_img = Image.open(str('C:/Users/μ΅μ¬μ/Desktop/main.png'))
st.image(main_img, width = 700)
#
# st.subheader('μκ°κΈ')
# st.write('μκ°κΈ')

