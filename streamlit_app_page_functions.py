import streamlit as st
from streamlit_app_functions.theoretical_description import text as text_theoretical_description
print("Page functions loaded.")

def header():
    st.title('Magnetization dynamics for FM/HM interfaces, a single-spin model')
    st.header('Online LLG integrator')
    st.caption("Joshua Salazar, S. Koraltan, C. Abert, P. Flauger, M. Agrawal, S. Zeilinger, A. Satz, C. Schmitt, G. Jakob, R. Gupta, M. Kläui, H. Brückl, J. Güttinger and Dieter Suess")
    st.caption("University of Vienna - Physics of Functional Materials")

def text_description():
    st.subheader('Theoretical description')
    text_theoretical_description()
