import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from phik import phik_matrix


def run():
    st.header("Welcome to the Data Analysis page!")
    st.write("Here is a simple Exploratory Data Analysis study, followed by relevant visualizations.")

    # import visualisasi
    main_data = pd.read_csv('online_shoppers_intention.csv')

    st.write("\n")
    st.write("##### 1. The dataset used in this study")
    st.write("The purpose of this project is to classify which web visitors will likely to make a purchase on a website based on the information kept in the provided dataset. The dataset contains 10 numerical and 8 categorical attributes, in which the target attribute will be Revenue as it represents whether the web visitor proceeds to payment (True) or not (False).")
    st.dataframe(main_data)

    # 1. for correlation matrix
    # make a new dataframe containing only numerical and boolean data
    data_corr_df = main_data.drop(['VisitorType','Month'], axis = 1)

    # create correlation matrix from the dataset
    correlation_matrix = data_corr_df.corr()

    st.write("\n")
    st.write("##### 1. Pearson Correlation Matrix Result")
    st.write("This is for observing linear relationship among numerical data with each other and the target, Revenue.")

    # # make a correlation matrix
    plt.figure(figsize=(10, 8))
    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='Pastel1', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    # display
    st.pyplot(heatmap.figure)

    # 2. make phi-k correlation heatmap

    # make a new dataframe
    matrix_df = main_data[['Month','VisitorType','Weekend','Revenue']]

    # Insert the data into phik_matrix calculation
    phi_k_corr = phik_matrix(matrix_df)

    st.write("\n")
    st.write("##### 2. Phik-K Correlation Heatmap Result")
    st.write("This is for checking relationship between categorical data with Revenue.")

    # Show the heatmap
    plt.figure(figsize=(10, 8))
    heatmap_phik = sns.heatmap(phi_k_corr, annot=True, cmap='Pastel1', fmt='.2f')
    plt.title('Phi-K Correlation Heatmap')
    # display
    st.pyplot(heatmap_phik.figure)


    # 3. bar plot

    # Between Weekend and Revenue
    t_revenue_weekend = main_data[main_data['Weekend'] == True]['Revenue'].value_counts()
    f_revenue_weekend = main_data[main_data['Weekend'] == False]['Revenue'].value_counts()

    # Create x-axis positions
    x_pos = [0, 1]

    # Create bar plot
    fig3, ax = plt.subplots(figsize=(10, 8))  # Create a figure and axis object
    ax.bar([x_pos[0], x_pos[1]], t_revenue_weekend, width=0.2, label='Weekend',color='lightblue')
    ax.bar([x_pos[0] + 0.2, x_pos[1] + 0.2], f_revenue_weekend, width=0.2, label='Weekday', color='salmon')

    # Adjust x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['True', 'False'])

    # Set labels and title
    ax.set_xlabel('Revenue')
    ax.set_ylabel('Count')
    ax.set_title('Number of True Cases for Revenue Based on Weekend')
    ax.legend()

    st.write("\n")
    st.write("##### 3. Number of Successful Transactions in Weekend (True) vs Weekday (False)")
    st.write("This is for comparing the difference in number of successful and failed transactions based on day type (weekend or weekday).")

    # Display the plot in Streamlit
    st.pyplot(fig3)  # Pass the figure object to st.pyplot()

if __name__ == '__main__':
    run()