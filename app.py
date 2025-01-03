import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
data = load_iris()
iris_df = pd.DataFrame(data=data.data, columns=data.feature_names)
iris_df['species'] = pd.Categorical.from_codes(data.target, data.target_names)

# Streamlit app
def main():
    st.title("Iris Dataset Visualization")
    st.sidebar.title("Visualization Options")

    # Sidebar options
    options = ["Dataset Overview", "Summary Statistics", "Visualizations"]
    choice = st.sidebar.selectbox("Select an option:", options)

    if choice == "Dataset Overview":
        st.header("Dataset Overview")
        st.write(iris_df.head())
        st.write(f"Shape of dataset: {iris_df.shape}")

    elif choice == "Summary Statistics":
        st.header("Summary Statistics")
        st.write(iris_df.describe())

    elif choice == "Visualizations":
        st.header("Visualizations")

        # Visualization options
        plot_type = st.sidebar.selectbox(
            "Select plot type:", ["Scatter Plot", "Histogram", "Pair Plot"]
        )

        if plot_type == "Scatter Plot":
            x_axis = st.sidebar.selectbox("Select X-axis:", iris_df.columns[:-1])
            y_axis = st.sidebar.selectbox("Select Y-axis:", iris_df.columns[:-1])

            st.write(f"Scatter Plot: {x_axis} vs {y_axis}")
            fig, ax = plt.subplots()
            sns.scatterplot(data=iris_df, x=x_axis, y=y_axis, hue="species", ax=ax)
            st.pyplot(fig)

        elif plot_type == "Histogram":
            column = st.sidebar.selectbox("Select column:", iris_df.columns[:-1])

            st.write(f"Histogram: {column}")
            fig, ax = plt.subplots()
            sns.histplot(data=iris_df, x=column, hue="species", kde=True, ax=ax)
            st.pyplot(fig)

        elif plot_type == "Pair Plot":
            st.write("Pair Plot")
            pair_plot_hue = st.sidebar.selectbox("Hue (Species):", [None, "species"])
            fig = sns.pairplot(iris_df, hue=pair_plot_hue)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
