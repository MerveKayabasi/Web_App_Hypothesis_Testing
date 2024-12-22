import streamlit as st
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene
import io

# Page Title
st.title('Hypothesis Testing and Statistical Analysis Application')

# Description
st.write("This application allows you to perform hypothesis testing and various statistical analyses on the uploaded dataset or manual input.")

# Data Input Option
input_method = st.radio("Select data input method:", ["Upload CSV File", "Enter Data Manually"])

if input_method == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
else:
    manual_input = st.text_area("Enter data (comma-separated or space-separated):")
    if manual_input:
        try:
            data = pd.read_csv(io.StringIO(manual_input), sep="[ ,]+", engine='python', header=None)
            data.columns = [f'Column_{i+1}' for i in range(data.shape[1])]
        except Exception as e:
            st.error(f"Error parsing data: {e}")
            data = None

if 'data' in locals() and data is not None:
    st.write("Dataset:")
    st.write(data.head())
    
    # Column Selection
    columns = data.columns.tolist()
    selected_column = st.selectbox("Which column should we test?", columns)
    
    # Display Basic Statistics
    st.write("Basic Statistics for Selected Column:")
    st.write(data[selected_column].describe())
    
    # Data Visualization
    st.write("Data Distribution:")
    fig, ax = plt.subplots()
    sns.histplot(data[selected_column], kde=True, ax=ax)
    st.pyplot(fig)
    
    # Normality and Homogeneity of Variance Tests
    st.write("**Normality and Homogeneity of Variance Tests**")
    normal_test_stat, normal_test_p = shapiro(data[selected_column])
    st.write(f"Shapiro-Wilk Test (Normality): p-value = {normal_test_p:.4f}")
    
    homogeneity_test_stat, homogeneity_test_p = levene(data[selected_column], data[selected_column])
    st.write(f"Levene Test (Homogeneity of Variance): p-value = {homogeneity_test_p:.4f}")

    if normal_test_p > 0.05 and homogeneity_test_p > 0.05:
        st.success("Parametric tests can be applied.")
        test_category = "Parametric"
    else:
        st.warning("Non-parametric tests are recommended.")
        test_category = "Non-parametric"

    # Hypothesis Test Input
    test_value = st.number_input("Enter the value to compare:", value=0.0)
    
    if test_category == "Parametric":
        test_type = st.selectbox("Select Parametric Test", ["One-sample t-test", "Independent two-sample t-test", "Paired t-test", "ANOVA"])
    else:
        test_type = st.selectbox("Select Non-parametric Test", ["Mann-Whitney U Test", "Wilcoxon Test", "Kruskal-Wallis Test"])
    
    if st.button("Perform Hypothesis Test"):
        sample = data[selected_column]
        
        if test_type == "One-sample t-test":
            t_stat, p_value = stats.ttest_1samp(sample, test_value)
        elif test_type == "Independent two-sample t-test":
            t_stat, p_value = stats.ttest_ind(sample, data[selected_column])
        elif test_type == "Paired t-test":
            t_stat, p_value = stats.ttest_rel(sample, data[selected_column])
        elif test_type == "ANOVA":
            groups = st.multiselect("Select groups for analysis:", columns)
            samples = [data[group] for group in groups if group in data]
            f_stat, p_value = stats.f_oneway(*samples)
            t_stat = f_stat
        elif test_type == "Mann-Whitney U Test":
            u_stat, p_value = stats.mannwhitneyu(sample, data[selected_column])
            t_stat = u_stat
        elif test_type == "Wilcoxon Test":
            t_stat, p_value = stats.wilcoxon(sample - test_value)
        elif test_type == "Kruskal-Wallis Test":
            groups = st.multiselect("Select groups for analysis:", columns)
            samples = [data[group] for group in groups if group in data]
            h_stat, p_value = stats.kruskal(*samples)
            t_stat = h_stat
        
        st.write("Test Statistic:", t_stat)
        st.write("p-value:", p_value)
        
        if p_value < 0.05:
            st.success("Result: The null hypothesis is rejected. The sample mean is different.")
        else:
            st.warning("Result: The null hypothesis cannot be rejected. The sample mean is not different.")
