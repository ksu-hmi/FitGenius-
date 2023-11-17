import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

def regression(df, col1, col2):
    """
    A function that takes a dataframe and the name of two columns 
    returns a linear regression using the two columns 
    and plots the result directly.
    """
    x = df[col1].values
    y = df[col2].values

    x = x.reshape(len(df), 1)
    y = y.reshape(len(df), 1)    

    regr = linear_model.LinearRegression()
    regr.fit(x, y)

    plt.scatter(x, y, color='red')

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.scatter(x, regr.predict(x), color='blue')

    st.pyplot(fig)

def kmeans(df, col1, col2, k):  
    """
    A function that takes a dataframe and the name of two columns, as well as K value
    computes a kmeans using the two columns with k = K that is passed in input
    and plots the result directly.
    """
    df = df.fillna(0)

    kmeans = KMeans(n_clusters=k).fit(df[[col1, col2]])
    centroids = kmeans.cluster_centers_
    st.write(centroids)

    fig, ax = plt.subplots()
    ax.scatter(df[col1], df[col2], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)

    st.pyplot(fig)

def display_gen(df):
    """
    A function that takes a dataframe and computes and prints in the Streamlit interface
    different metrics.
    """
    with st.expander("Data"):
        st.write(df)

    st.markdown("<h2 style='text-align: center;'>General Figures</h2>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric('Number of Category 1', int(len(df[df['Heart Attack Risk'] == 1])))
    col2.metric('Number of Category 2', int(len(df[df['Heart Attack Risk'] == 2])))
    col3.metric('Number of Category 3', int(len(df[df['Heart Attack Risk'] == 3])))
    col4.metric('Number of Category 4', int(len(df[df['Heart Attack Risk'] == 4])))

    st.markdown("<h2 style='text-align: center;'>Distributions</h2>", unsafe_allow_html=True)

    fig = px.histogram(df, x="Heart Attack Risk", color="Age", marginal="violin", hover_data=df.columns)

    col1, col2 = st.columns([1, 1])
    col1.plotly_chart(fig, use_container_width=True)

    df_agg = df.groupby(['Heart Attack Risk'], as_index=False)['Age'].mean()
    col2.write("Heart Attack Risk")
    col2.table(df_agg)

    st.markdown("<h2 style='text-align: center;'>Aggregation by column</h2>", unsafe_allow_html=True)

    possible_rows = df.columns
    col1, col2, col3 = st.columns([1, 1, 1])
    x_axis_select = col1.selectbox("X-axis", possible_rows[1:], index=1)
    color_select = col2.selectbox("Color", possible_rows[1:], index=2)
    barmode = col3.selectbox("Bar Mode", ['stack', 'group'])

    df_grp = df.groupby(by=[x_axis_select, color_select]).size().to_frame('size')
    df_grp = df_grp.reset_index()
    fig = px.bar(df_grp, x=x_axis_select, y='size', color=color_select, barmode=barmode)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h2 style='text-align: center;'>Average over column</h2>", unsafe_allow_html=True)
    if is_numeric_dtype(df[color_select]):
        df_grp = df.groupby(by=[x_axis_select], as_index=False)[color_select].mean()
        fig = px.line(df_grp, x=x_axis_select, y=color_select)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning('A non-numeric column, cannot compute the average')
def ml_interface(df):
    st.markdown("<h2 style='text-align: center;'>Machine Learning Interface</h2>", unsafe_allow_html=True)

    # Assume 'Heart Attack Risk' is your target variable (modify as needed)
    target_variable = 'Heart Attack Risk'
    
    # Features and target variable
    features = ['Alcohol Consumption', 'Age', 'Heart Rate', 'Physical Activity Days Per Week']
    X = df[features]
    y = df[target_variable]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Machine Learning model (Random Forest Classifier)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Predictions on the test set
    y_pred = clf.predict(X_test)

    # Display evaluation metrics
    st.subheader('Classification Report:')
    st.text(classification_report(y_test, y_pred))

    st.subheader('Accuracy Score:')
    st.text(accuracy_score(y_test, y_pred))

    # Feature Importances
    st.subheader('Feature Importances:')
    feature_importance = pd.DataFrame(clf.feature_importances_, index=features, columns=['Importance'])
    st.bar_chart(feature_importance)

# Main render function
def render():
    if 'df_health' not in st.session_state:
        # Load your health dataset (replace 'your_dataset.csv' with the actual file name)
        df = pd.read_csv('heart_health.csv')  
        st.session_state['df_health'] = df
    else:
        df = st.session_state['df_health']

    options = st.sidebar.selectbox('Mode', ("Display", "Kmeans", "Regression", "Health and Fitness", "Machine Learning"))

    if options == 'Display':    
        display_gen(df)
    elif options == "Kmeans":
        possible_rows = df.columns

        fig = px.scatter_matrix(df,
            dimensions=['Age', 'Alcohol Consumption', 'Heart Rate', 'Physical Activity Days Per Week'],
            color="Heart Attack Risk", symbol="Heart Attack Risk",
            title="Scatter matrix",
            labels={col: col.replace('_', ' ') for col in df.columns})  # remove underscore
        config = {
            'toImageButtonOptions': {
                'format': 'png',  # one of png, svg, jpeg, webp
                'filename': 'scatter_matrix',
                'height': 500,
                'width': 2000,
                'scale': 5  # Multiply title/legend/axis/canvas sizes by this factor
            }
        }
        fig.update_traces(diagonal_visible=False)
        fig.update_layout(height=1000)
        st.plotly_chart(fig, use_container_width=True, config=config, height=1000)

        col1, col2, col3, col4 = st.columns(4)
        x_axis_select = col1.selectbox("X-axis", possible_rows[1:])
        y_axis_select = col2.selectbox("Y-axis", possible_rows[1:])

        k_value = col3.number_input("K-Value", value=3)
        col4.write("")
        btn_load = col4.button("Load")
        if btn_load:
            kmeans(df, x_axis_select, y_axis_select, k_value)

    elif options == "Regression":
        possible_rows = df.columns

        fig = px.scatter_matrix(df,
            dimensions=['Age', 'Alcohol Consumption', 'Heart Rate', 'Physical Activity Days Per Week'],
            color="Heart Attack Risk", symbol="Heart Attack Risk",
            title="Scatter matrix",
            labels={col: col.replace('_', ' ') for col in df.columns})  # remove underscore
        config = {
            'toImageButtonOptions': {
                'format': 'png',  # one of png, svg, jpeg, webp
                'filename': 'scatter_matrix',
                'height': 500,
                'width': 2000,
                'scale': 5  # Multiply title/legend/axis/canvas sizes by this factor
            }
        }
        fig.update_traces(diagonal_visible=False)
        fig.update_layout(height=1000)
        st.plotly_chart(fig, use_container_width=True, config=config, height=1000)

        col1, col2, col3 = st.columns(3)
        x_axis_select = col1.selectbox("X-axis", possible_rows[1:])
        y_axis_select = col2.selectbox("Y-axis", possible_rows[1:])
        btn_load = col3.button("Load")
        if btn_load:
            regression(df, x_axis_select, y_axis_select)

    elif options == "Health and Fitness":
        # Add your health and fitness logic here
        st.write("Health and Fitness Mode")

    elif options == "Machine Learning":
        ml_interface(df)

# Call the render function
render()

