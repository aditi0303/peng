# Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'penguin_app.py'.

# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame

df = pd.read_csv("penguin.csv")

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)


#ACT:2
def prediction(model, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex):
    # Create a dictionary to map species index to species name
    species_mapping = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}
    
    # Prepare the input data for prediction
    input_data = [[island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex]]
    
    # Call the predict() function for classification based on the model object
    predicted_species_index = model.predict(input_data)[0]
    
    # Extract the integer value using indexing
    species_name = species_mapping[predicted_species_index]
    
    # Return the name of the species
    return species_name

#ACT:3

st.title("Penguin Species Prediction App")
st.sidebar.header("Input Features")
bill_length = st.slider("Bill Length (mm)", float(X['bill_length_mm'].min()), float(X['bill_length_mm'].max()), float(X['bill_length_mm'].mean()))
bill_depth = st.slider("Bill Depth (mm)", float(X['bill_depth_mm'].min()), float(X['bill_depth_mm'].max()), float(X['bill_depth_mm'].mean()))
flipper_length = st.slider("Flipper Length (mm)", float(X['flipper_length_mm'].min()), float(X['flipper_length_mm'].max()), float(X['flipper_length_mm'].mean()))
body_mass = st.slider("Body Mass (g)", float(X['body_mass_g'].min()), float(X['body_mass_g'].max()), float(X['body_mass_g'].mean()))

sex = st.sidebar.selectbox("Sex", ['Male', 'Female'])
island = st.sidebar.selectbox("Island", ['Biscoe', 'Dream', 'Torgersen'])

classifier = st.sidebar.selectbox("Classifier", ['Random Forest', 'SVM', 'Logistic Regression'])

# Predict button
if st.sidebar.button("Predict"):
    if classifier == 'Random Forest':
        model = rf_clf
    elif classifier == 'SVM':
        model = svc_model
    elif classifier == 'Logistic Regression':
        model = log_reg

    prediction = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, model.predict(X_test))
    st.write("Predicted Species:", prediction[0])
    st.write("Accuracy Score:", accuracy)

