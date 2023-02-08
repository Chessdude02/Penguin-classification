# Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'penguin_app.py'.

# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

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

def prediction(model,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex):
	species=model.predict([[island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex]])
	species=species[0]
	if species==0:
		return" Adelie"
	elif species==2:
		return" Chinstrap"
	else:
		return" Gentoo"

st.sidebar.title("Penguin prediction App")
isl=st.sidebar.slider("Island",0,2)
bill_le=st.sidebar.slider("Bill_length_mm",float(df['bill_length_mm'].min()),float(df['bill_length_mm'].max()))
bill_de=st.sidebar.slider("Bill_depth_mm",float(df['bill_depth_mm'].min()),float(df['bill_depth_mm'].max()))
flipper_le=st.sidebar.slider("Flipper_length_mm",float(df['flipper_length_mm'].min()),float(df['flipper_length_mm'].max()))
body_ma=st.sidebar.slider("Body_mass_g",float(df['body_mass_g'].min()),float(df['body_mass_g'].max()))
sex=st.sidebar.slider("Sex",0,1)
classifier = st.sidebar.selectbox('Classifier', ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))
if st.sidebar.button("Predict"):
	if classifier =='Suport Vector Mchine':
		species_type=prediction(svc_model,isl,bill_le,bill_de,flipper_le,body_ma,sex)
		score=svc_model.score(X_train,y_train)
	elif classifier=='LogisticRegression':
		species_type=prediction(log_reg,isl,bill_le,bill_de,flipper_le,body_ma,sex)
		score=log_reg.score(X_train,y_train)
	else:
		species_type=prediction(rf_clf,isl,bill_le,bill_de,flipper_le,body_ma,sex)
		score=rf_clf.score(X_train,y_train)
	st.write("Species_predicted=",species_type)
	st.write("Accuracy of model is:",score)
