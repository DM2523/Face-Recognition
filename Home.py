import streamlit as st

st.set_page_config(page_title = 'Home',
                   layout='wide',
                   initial_sidebar_state='expanded')
st.title('Facial recognition system')
st.caption('This webapp uses (Yolov8s+Haar Cascade) for face detection and Custom Facenet for recognition.')

st.markdown("""
### Add a new person and check.
You can add new perosn to [backend](/Add) and Test it [here](/Recognition).
""")