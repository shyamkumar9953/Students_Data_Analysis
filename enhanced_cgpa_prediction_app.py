# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel, chi2_contingency, f_oneway, wilcoxon, mannwhitneyu, kruskal, friedmanchisquare

# Add Delhi Technological University Logo
st.image("dtu_logo.png", width=200);
st.title("CGPA Prediction and Analysis Tool")
st.write("Created by:Shyam kumar(Roll No: 2K22/SE/171)")
st.write("Delhi Technological University")

# Load and preprocess data
@st.cache_data
def preprocess_data(data):
    imputer = SimpleImputer(strategy="mean")
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Encode categorical features
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = LabelEncoder().fit_transform(data[column])
    
    # Split data
    X = data.drop('CGPA', axis=1)
    y = data['CGPA']
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

# Load dataset
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    X, y = preprocess_data(data)

    # Show data preview
    st.write("Data Preview", data.head())

    # Split data into train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model selection
    st.sidebar.title("Choose Model")
    model_type = st.sidebar.selectbox("Select Model", ("K-Nearest Neighbors", "Support Vector Regression", "Random Forest"))
    
    if model_type == "K-Nearest Neighbors":
        model = KNeighborsRegressor()
    elif model_type == "Support Vector Regression":
        model = SVR()
    else:
        model = RandomForestRegressor()

    # Train the model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Model evaluation
    st.subheader("Model Evaluation")
    st.write("Mean Squared Error:", mean_squared_error(y_test, predictions))
    st.write("R-squared Score:", r2_score(y_test, predictions))

    # Statistical tests
    st.sidebar.title("Statistical Tests")
    test_type = st.sidebar.selectbox("Choose Statistical Test", ("One-sample t-test", "Two-sample t-test", "Paired t-test", "Chi-square test", "F-test", "One-way ANOVA", "Two-way ANOVA", "Wilcoxon rank-sum test", "Mann-Whitney U test", "Kruskal-Wallis test", "Friedman test"))
    
    # Perform selected statistical test
    if test_type == "One-sample t-test":
        t_stat, p_val = ttest_1samp(y_test - predictions, 0)
    elif test_type == "Two-sample t-test":
        t_stat, p_val = ttest_ind(y_test, predictions)
    elif test_type == "Paired t-test":
        t_stat, p_val = ttest_rel(y_test, predictions)
    elif test_type == "Chi-square test":
        chi2, p_val, _, _ = chi2_contingency(pd.crosstab(y_test, np.round(predictions)))
        t_stat = chi2
    elif test_type == "F-test":
        t_stat, p_val = f_oneway(y_test, predictions)
    elif test_type == "One-way ANOVA":
        t_stat, p_val = f_oneway(y_test, predictions)
    elif test_type == "Wilcoxon rank-sum test":
        t_stat, p_val = wilcoxon(y_test - predictions)
    elif test_type == "Mann-Whitney U test":
        t_stat, p_val = mannwhitneyu(y_test, predictions)
    elif test_type == "Kruskal-Wallis test":
        t_stat, p_val = kruskal(y_test, predictions)
    elif test_type == "Friedman test":
        t_stat, p_val = friedmanchisquare(y_test, predictions, y_train[:len(y_test)])

    # Display test results
    st.subheader("Statistical Test Results")
    st.write(f"Test Statistic: {t_stat:.4f}")
    st.write(f"P-value: {p_val:.4f}")
    
    # Cross-validation for model validation
    st.subheader("Model Validation")
    cv_score = cross_val_score(model, X, y, cv=5, scoring='r2')
    st.write("Cross-validated R-squared:", np.mean(cv_score))
else:
    st.info("Awaiting file upload...")

# Conclusion
st.write("This app allows model training, evaluation, and statistical testing on CGPA prediction data.")
