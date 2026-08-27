import streamlit as st

def load_css(file_path: str = "assets/styles.css"):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)