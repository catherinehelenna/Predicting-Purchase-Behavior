import streamlit as st
import pandas as pd
import pickle
from PIL import Image

def run():
    # buat header
    st.header("Welcome to the Prediction page!")

    # description
    st.write("This is the model evaluation section.")
    st.write("The confusion matrix illustrates the number of correct and incorrect predictions regarding whether web visitors will complete a purchase or not.")
    st.write("For forecasting web visitor purchase decisions, it is crucial to lower false negatives (i.e., fail to recognize visitors who will likely make a purchase) to ensure effective targeting of marketing efforts and generate more revenue. Achieving a high recall (above 0.5) indicates that the model seamlessly captures a large proportion of visitors who are likely to make a purchase, reducing missed opportunities for customer conversion.")

    # Load image
    st.write('\n')
    gambar = Image.open('confusion_matrix.png')
    st.image(gambar)
    st.write("Accuracy = 88.6%, Recall = 77%.")


    st.write("Please input the web visitor information.")

    with open('model.pkl','rb') as file_1:
        my_model = pickle.load(file_1)

    # for capping
    with open('capping.pkl', 'rb') as file_2:
        capping = pickle.load(file_2)

    # for PCA
    with open('pca_function.pkl', 'rb') as file_3:
        pca_fix = pickle.load(file_3)

    # for pipeline
    with open('pipeline.pkl', 'rb') as file_4:
        pipe_line = pickle.load(file_4)

#     # input all stuffs about customer
    with st.form("Input customer information"):
        admin = st.slider('The accessed administration page number',min_value =0,max_value = 27,step=1)
        info = st.slider('The accessed information page number',min_value =0,max_value = 27,step=1)
        product = st.slider('The accessed product related page number',min_value =0,max_value = 705,step = 1)
        bouncerate = st.number_input('Bounce rate: the proportion of visitors only viewing a page from 0 to 1',step=0.01)
        exitrate = st.number_input('Exit rate: the proportion of visitors who leave a specific page on a website after viewing it as the last page in their session from 0 to 1.',step = 0.01)
        page_values = st.slider('Page values: the value of individual pages on a website in terms of their contribution to goal completions or conversions', min_value = 0,max_value = 362)
        month = st.selectbox('Choose the month of transaction', ['Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        browser = st.slider('Browser',min_value = 1, max_value = 13,step =1)
        traffictype = st.slider('Traffic Type',min_value = 1, max_value = 20,step =1)
        visitor_type = st.selectbox('Choose type of web visitor', ['Returning_Visitor', 'New_Visitor', 'Other'])
        # submit form
        sub = st.form_submit_button('Predict')

    if sub:
        data_predict = {'Administrative': admin,
                        'Informational':info,'ProductRelated':product,
                        'BounceRates':bouncerate,'ExitRates':exitrate,
                        'PageValues':page_values,'Month':month,
                        'Browser':browser,'TrafficType':traffictype,'VisitorType':visitor_type}
        
        data = pd.DataFrame([data_predict])
        st.write("#### Input data result:")
        st.dataframe(data)
        try:
            data_capped = capping.transform(data)

            # preprocess and encoding categorical data
            X_inf_encoded_scaled = pipe_line.transform(data_capped)

            # return back to dataframe
            column = ['Month_Aug', 'Month_Dec', 'Month_Feb', 'Month_Jul', 'Month_June', 'Month_Mar', 'Month_May', 'Month_Nov', 'Month_Oct', 'Month_Sep', 'VisitorType_New_Visitor', 'VisitorType_Other', 'VisitorType_Returning_Visitor', 'Administrative', 'Informational', 'ProductRelated', 'BounceRates', 'ExitRates', 'PageValues', 'Browser', 'TrafficType']

            # make dataframe
            X_inf_encoded_scaled_final = pd.DataFrame(X_inf_encoded_scaled, columns=column)

            # prepare for PCA
            X_inf_for_pca = X_inf_encoded_scaled_final.iloc[:, 12:]

            # drop for replacing with PCA
            X_inf_encoded_scaled_pca_final = X_inf_encoded_scaled_final.drop(X_inf_encoded_scaled_final.columns[12:], axis=1)

            # PCA
            X_inf_scaled_pca = pca_fix.transform(X_inf_for_pca)

            # make back into dataframe
            pca_inf_df = pd.DataFrame(X_inf_scaled_pca, columns=[f"PC{i+1}" for i in range(X_inf_scaled_pca.shape[1])])

            # concantenate all data into one dataframe
            X_inf_combined = pd.concat([X_inf_encoded_scaled_pca_final , pca_inf_df], axis=1)

            # make prediction
            predictions = my_model.predict(X_inf_combined)

            # convert prediction
            converted_predictions = [bool(pred) for pred in predictions]

            st.write('Predicted web-visitor decision to complete the transaction:', converted_predictions[0])
        except Exception as e:
            st.error(f'An error occurred: {e}')
    
            
if __name__ == '__main__':
    run()
