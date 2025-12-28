import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def eda_univariate(df:pd.DataFrame):
    figs = []
    fig, ax = plt.subplots()
    sns.histplot(df["charges"], kde=True, ax=ax)
    ax.set_title("Distribution of Medical Charges")
    figs.append(fig)

    fig, ax = plt.subplots()
    sns.histplot(df["age"], bins=20, kde=False, ax=ax)
    ax.set_title("Age Distribution")
    figs.append(fig)

    fig, ax = plt.subplots()
    sns.countplot(x="smoker", data=df, ax=ax)
    ax.set_title("Smokers vs Non-smokers")
    figs.append(fig)

    fig, ax = plt.subplots()
    sns.histplot(df["bmi"], kde=True, ax=ax)
    ax.set_title("BMI Distribution")
    figs.append(fig)

    fig, ax = plt.subplots()
    sns.countplot(x="region", data=df, ax=ax)
    ax.set_title("Region-wise Policyholder Count")
    figs.append(fig)
    return figs

def eda_bivariate(df:pd.DataFrame):
    figs = []
    fig, ax = plt.subplots()
    sns.scatterplot(x="age", y="charges", hue="smoker", data=df, ax=ax)
    ax.set_title("Charges vs Age (colored by smoker)")
    figs.append(fig)

    fig, ax = plt.subplots()
    sns.boxplot(x="smoker", y="charges", data=df, ax=ax)
    ax.set_title("Charges by Smoking Status")
    figs.append(fig)

    fig, ax = plt.subplots()
    sns.scatterplot(x="bmi", y="charges", hue="sex", data=df, ax=ax)
    ax.set_title("Charges vs BMI")
    figs.append(fig)

    fig, ax = plt.subplots()
    sns.boxplot(x="sex", y="charges", data=df, ax=ax)
    ax.set_title("Charges by Gender")
    figs.append(fig)

    fig, ax = plt.subplots()
    sns.boxplot(x="children", y="charges", data=df, ax=ax)
    ax.set_title("Charges by Number of Children")
    figs.append(fig)
    return figs

def eda_multivariate(df:pd.DataFrame):
    figs = []
    fig, ax = plt.subplots()
    sns.scatterplot(x="age", y="charges", hue="smoker", style="sex", data=df, ax=ax)
    ax.set_title("Smoking Status and Age vs Charges")
    figs.append(fig)

    fig, ax = plt.subplots()
    sns.boxplot(x="region", y="charges", hue="sex", data=df[df["smoker"]=="yes"], ax=ax)
    ax.set_title("Gender and Region on Charges for Smokers")
    figs.append(fig)

    fig, ax = plt.subplots()
    sns.scatterplot(x="age", y="charges", hue="smoker", size="bmi", data=df, ax=ax)
    ax.set_title("Age, BMI, Smoking Status vs Charges")
    figs.append(fig)

    df2 = df.copy()
    df2["obese"] = (df2["bmi"] > 30).astype(int)
    df2["group"] = df2.apply(lambda r: "Obese Smoker" if r["obese"]==1 and r["smoker"]=="yes"
                             else "Non-obese Non-smoker" if r["obese"]==0 and r["smoker"]=="no"
                             else "Other", axis=1)
    fig, ax = plt.subplots()
    sns.boxplot(x="group", y="charges", data=df2, ax=ax)
    ax.set_title("Obese Smokers vs Non-obese Non-smokers")
    figs.append(fig)
    return figs

def eda_outliers_corr(df:pd.DataFrame):
    figs = []
    fig, ax = plt.subplots()
    sns.boxplot(y="charges", data=df, ax=ax)
    ax.set_title("Outliers in Charges")
    figs.append(fig)

    fig, ax = plt.subplots()
    sns.boxplot(y="bmi", data=df, ax=ax)
    ax.set_title("Outliers in BMI")
    figs.append(fig)

    fig, ax = plt.subplots(figsize=(6,5))
    corr = df[["age","bmi","children","charges"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    figs.append(fig)
    return figs