import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def load_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    return data

def exploratory_data_analysis(data):
    # Basic information about the dataset
    st.subheader("Dataset Information")
    st.write(data.info())

    st.subheader("Sample Data")
    st.write(data.sample(10))

    st.subheader("total number of missing values")
    missing_values = data.isnull().sum().sort_values(ascending=False)
    total_rows = len(data)
    missing_values_percentage = (missing_values / total_rows) * 100
    st.write(missing_values)
    
    # Missing values
    st.subheader("Missing Values by percentage")
    missing_values = data.isnull().sum()
    total_rows = len(data)
    missing_values_percentage = (missing_values / total_rows) * 100
    st.write(missing_values_percentage[missing_values_percentage > 0])


    # Data types
    st.subheader("Data Types")
    st.write(data.dtypes)

    # Descriptive statistics
    st.subheader("Descriptive Statistics")
    st.write(data.describe())

    # Outliers
    st.subheader("Outliers")
    
    plot_type = st.radio("Select Plot Type for Outlier Detection", ["Box Plot", "Scatter Plot", "Histogram"])
    column = st.selectbox("Select Column for Plotting", data.select_dtypes(include=[np.number]).columns)

    if plot_type == "Box Plot":
        fig, ax = plt.subplots()
        sns.boxplot(data=data, x=column, ax=ax)
        st.pyplot(fig)
    elif plot_type == "Scatter Plot":
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x=column, y=column, ax=ax)
        st.pyplot(fig)
    elif plot_type == "Histogram":
        fig, ax = plt.subplots()
        sns.histplot(data[column], ax=ax)
        st.pyplot(fig)

    # Visualizations
    st.subheader("Visualizations")

    try:
        visualization_type = st.radio("Select Visualization Type", ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Pie Chart", "Box Plot", "Heatmap"])
        column = st.selectbox("Select Column to Visualize", data.select_dtypes(include=[np.number]).columns)

        if visualization_type == "Scatter Plot":
            fig, ax = plt.subplots()
            sns.scatterplot(data=data, x=data.columns[0], y=column, ax=ax)
            st.pyplot(fig)
        elif visualization_type == "Line Chart":
            fig, ax = plt.subplots()
            data.plot(y=column, ax=ax, kind="line")
            st.pyplot(fig)
        elif visualization_type == "Bar Chart":
            fig, ax = plt.subplots()
            data[column].plot(ax=ax, kind="bar")
            st.pyplot(fig)
        elif visualization_type == "Histogram":
            fig, ax = plt.subplots()
            sns.histplot(data[column], ax=ax)
            st.pyplot(fig)
        elif visualization_type == "Pie Chart":
            fig, ax = plt.subplots()
            data[column].value_counts().plot(ax=ax, kind="pie", autopct="%1.1f%%")
            st.pyplot(fig)
        elif visualization_type == "Box Plot":
            fig, ax = plt.subplots()
            sns.boxplot(data=data, x=column, ax=ax)
            st.pyplot(fig)
        elif visualization_type == "Heatmap":
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

    except KeyError:
        st.error("The selected column does not exist in the dataset.")
    except ValueError:
        st.error("The selected column cannot be visualized.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    # Correlation
    st.subheader("Correlation")
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    corr_matrix = data[numeric_columns].corr()
    st.write(corr_matrix)

    # Correlation heatmap
    st.write("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

def generate_summary(data):
    summary = f"This dataset contains {len(data)} rows and {len(data.columns)} columns.\n\n"
    summary += "The columns in the dataset are:\n"
    for column in data.columns:
        summary += f"- {column}\n"
    summary += "\nSome key insights from the Exploratory Data Analysis:\n"
    summary += "- " + ", ".join([f"{col} has {(data[col].isnull().sum() / len(data)) * 100:.2f}% missing values" for col in data.columns if data[col].isnull().sum() > 0]) + "\n"
    unique_dtypes = list(data.dtypes.unique())
    summary += "- The dataset contains the following data types: " + ", ".join(map(str, unique_dtypes)) + "\n"
    summary += "- The descriptive statistics provide an overview of the central tendency and dispersion of the numerical columns.\n"
    summary += "- The correlation analysis and heatmap highlight the relationships between the numerical columns.\n"
    summary += "- The visualizations, such as histograms, box plots, and scatter plots, offer insights into the distribution and potential outliers in the data.\n"
    return summary

def detect_outliers(data, column):
    outlier_method = st.radio("Select Outlier Detection Method", ["Univariate Outlier Detection", "Multivariate Outlier Detection", "Robust Statistics"])
    if outlier_method == "Univariate Outlier Detection":
        st.write("Univariate Outlier Detection")
        # Univariate Outlier Detection using Z-scores
        z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
        threshold = 3
        outliers = data[z_scores > threshold]
        st.write(outliers)

    elif outlier_method == "Multivariate Outlier Detection":
        st.write("Multivariate Outlier Detection")
        # Multivariate Outlier Detection using Minimum Covariance Determinant (MCD)
        from sklearn.covariance import EllipticEnvelope
        clf = EllipticEnvelope(contamination=0.1)
        clf.fit(data)
        outliers = clf.predict(data)
        st.write(outliers)

    elif outlier_method == "Robust Statistics":
        st.write("Robust Statistics")
         # Robust Statistics using Median Absolute Deviation (MAD)
        median = data[column].median()
        mad = np.median(np.abs(data[column] - median))
        threshold = 3
        modified_z_scores = 0.6745 * (data[column] - median) / mad
        outliers = data[np.abs(modified_z_scores) > threshold]
        st.write(outliers)

def main():
    st.title("Exploratory Data Analysis Report")
    file_path = st.file_uploader("Upload a dataset", type=["csv"])

    if file_path is not None:
        data = load_data(file_path)
        st.subheader("Summary")
        summary = generate_summary(data)
        st.write(summary)
        # detect_outliers(data, data.select_dtypes(include=[np.number]).columns[0])
        exploratory_data_analysis(data)
        
        

if __name__ == "__main__":
    main()