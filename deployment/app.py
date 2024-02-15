import streamlit as st
import prediction
import eda
from PIL import Image


page = st.sidebar.selectbox(label= 'Select Menu: ', options=['Home','Data Analysis','Predict Potential Customer'])
    
if page == 'Home':
    st.header("E-Commerce Forecasting: Predicting Purchase Behavior from Website Data")
    st.write("\n")
    st.write("This is a web app to illustrate the results of EDA (Exploratory Data Analysis) as well as predictions if web visitors will purchase something on web or not based on the behavioral data pattern. If you're curious about the features available in this web app, please check the menu on the left sidebar.")
    myimage = Image.open('customer_journey.png')
    st.image(myimage, width = 500)
elif page == 'Data Analysis':
    eda.run()
else:
    prediction.run()