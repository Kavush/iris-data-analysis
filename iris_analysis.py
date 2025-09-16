import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

try:
    # Load dataset
    iris = load_iris(as_frame=True)
    df = iris.frame
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    print(df.head())

    # Explore dataset
    print(df.info())
    print(df.isnull().sum())

    # Basic statistics
    print(df.describe())

    # Grouping by species
    print(df.groupby("species")["sepal length (cm)"].mean())

    # Visualizations
    sns.pairplot(df, hue="species")
    plt.savefig("pairplot.png")

    # Line chart
    df.groupby("species")["sepal length (cm)"].mean().plot(
        kind="line", marker="o", title="Average Sepal Length by Species"
    )
    plt.xlabel("Species")
    plt.ylabel("Sepal Length (cm)")
    plt.legend(["Sepal Length"])
    plt.savefig("line_chart.png")
    plt.close()

    # Bar chart
    df.groupby("species")["sepal length (cm)"].mean().plot(
        kind="bar", color=["#1f77b4", "#ff7f0e", "#2ca02c"], title="Average Sepal Length per Species"
    )
    plt.xlabel("Species")
    plt.ylabel("Average Sepal Length (cm)")
    plt.savefig("bar_chart.png")
    plt.close()

    # Histogram
    df["sepal length (cm)"].plot(
        kind="hist", bins=20, title="Sepal Length Distribution", color="skyblue", edgecolor="black"
    )
    plt.xlabel("Sepal Length (cm)")
    plt.savefig("histogram.png")
    plt.close()

    # Scatter plot
    df.plot(
        kind="scatter",
        x="petal length (cm)",
        y="petal width (cm)",
        c=pd.Categorical(df["species"]).codes,
        cmap="viridis",
        title="Petal Length vs Petal Width"
    )
    plt.savefig("scatter_plot.png")
    plt.close()

    print("✅ All plots saved successfully!")

except FileNotFoundError:
    print("❌ Error: Dataset file not found. Please check the file path.")
except pd.errors.EmptyDataError:
    print("❌ Error: The dataset file is empty.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
