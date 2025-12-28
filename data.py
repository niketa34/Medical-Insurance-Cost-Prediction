import pandas as pd

CATEGORICAL = ["sex", "smoker", "region"]
NUMERIC = ["age", "bmi", "children"]
TARGET = "charges"

def load_data(path:str) -> pd.DataFrame:
    df = pd.read_csv("C:/Users/ADMIN/Desktop/New folder/proj_3/medical_insurance.csv")
    return df

def clean_data(df:pd.DataFrame) -> pd.DataFrame:
    # Drop duplicates
    df = df.drop_duplicates().copy()
    # Strip whitespace in categorical
    for col in CATEGORICAL:
        df[col] = df[col].astype(str).str.strip().str.lower()
    # Validate ranges (basic)
    df = df[(df["age"] >= 0) & (df["bmi"] > 0) & (df["children"] >= 0)]
    # Drop rows with missing critical values
    df = df.dropna(subset=CATEGORICAL + NUMERIC + [TARGET])
    return df