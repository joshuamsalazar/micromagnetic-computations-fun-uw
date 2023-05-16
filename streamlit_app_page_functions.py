import streamlit as st
from streamlit_app_functions.theoretical_description import text as text_theoretical_description
print("Page functions loaded.")

def text_description():
    st.subheader('Theoretical description')
    text_theoretical_description()
