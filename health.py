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
from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import export_text: line removed as it is not used in code

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

    st.markdown("<h2 style='text-align: center;'>FitGenius</h2>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric('Average Step Count', round(df['Actual Steps'].mean()))
    col2.metric('Average Calories Burned', round(df['Total Calories'].mean()))
    col3.metric('Average Stress Level', round(df['Stress'].mean()))


    st.markdown("<h2 style='text-align: center;'>Distributions</h2>", unsafe_allow_html=True)

    fig = px.histogram(df, x="Stress", color="Actual Steps", marginal="violin", hover_data=df.columns)

    col1, col2 = st.columns([1, 1])
    col1.plotly_chart(fig, use_container_width=True)

    df_agg = df.groupby(['Stress'], as_index=False)['Actual Steps'].mean()
    col2.write("Steps")
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
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import export_text: line removed as it it not called in code

# Function to create and train the decision tree model
def train_decision_tree(df):
    # Choose features and target variable
    features = ['Active Calories', 'Resting Calories', 'Total Calories', 'Actual Steps', 'Stress']  # Adjust based on your feature names
    X = df[features]
    y = df['Stress']

    # Create and train the Decision Tree model
    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X, y)

    return dt_clf

# Function to make predictions and display recommendations
def display_recommendations(dt_model, input_data):
    features = ['Active Calories', 'Resting Calories', 'Total Calories', 'Actual Steps', 'Stress']

    # Convert input_data to DataFrame before selecting columns
    input_data_df = pd.DataFrame(input_data).transpose()
    input_data_df = input_data_df[features]

    prediction = dt_model.predict(np.array(input_data_df).reshape(1, -1))[0]

    # Display recommendations based on the predicted class
    if input_data_df['Stress'].values[0] > 23:
        # Provide recommendations for high-stress scenarios
        st.write("Recommendation: Consider incorporating activities like yoga and meditation to manage stress.")
    elif input_data_df['Stress'].values[0] <= 23:
        # Provide recommendations for stress levels below or equal to 23
        if input_data_df['Total Calories'].values[0] > 0:
            if input_data_df['Total Calories'].values[0] > 1000:
                st.write("Great job! Your stress levels are low, and you're burning a substantial number of calories.")
                st.write("Recommendation: Continue with your current fitness routine and consider adding variety.")
            else:
                st.write("Your stress levels are low, but you can still benefit from increasing physical activity.")
                st.write("Recommendation: Add more exercises to your routine, such as brisk walking or cycling.")
        else:
            st.write("Low stress level detected, but it seems there's no data for total calories.")
            st.write("Recommendation: Ensure to track and maintain a balance between physical activity and calories.")
    else:
        # Default recommendations
        st.write("Default Recommendation: ...")






# Main render function
def render():
    if 'df_health' not in st.session_state:
        # Load your health dataset (replace 'your_dataset.csv' with the actual file name)
        df = pd.read_csv('WatchData.csv')  
        st.session_state['df_health'] = df
    else:
        df = st.session_state['df_health']

    options = st.sidebar.selectbox('Mode', ("Display", "Kmeans", "Regression", "Recommendations"))

    if options == 'Display':    
        display_gen(df)
    elif options == "Kmeans":
        possible_rows = df.columns

        fig = px.scatter_matrix(df,
            dimensions=['Stress', 'Total Calories', 'Actual Steps'],
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
            dimensions=['Stress', 'Total Calories', 'Actual Steps'],
            color="Stress", symbol="Stress",
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

    elif options == "Recommendations":
        sample_data = df[['Active Calories', 'Resting Calories', 'Total Calories', 'Actual Steps', 'Stress']].iloc[0]
        # added brackets to column selection above to correct syntax error
        dt_model = train_decision_tree(df)
        # added above line to train model before predictions
        display_recommendations(dt_model, sample_data)
        # switched order of variables to line above to match function

# Call the render function
render()
