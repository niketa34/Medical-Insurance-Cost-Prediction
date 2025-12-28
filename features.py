import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import CATEGORICAL, NUMERIC

def add_feature_engineering(df:pd.DataFrame) -> pd.DataFrame:
    # BMI class
    def bmi_class(b):
        if b < 18.5: return "underweight"
        elif b < 25: return "normal"
        elif b < 30: return "overweight"
        else: return "obese"
    df = df.copy()
    df["bmi_class"] = df["bmi"].apply(bmi_class)
    # Interaction terms
    df["smoker_yes"] = (df["smoker"] == "yes").astype(int)
    df["age_smoker_interaction"] = df["age"] * df["smoker_yes"]
    return df

def build_preprocessor(include_bmi_class:bool=True) -> ColumnTransformer:
    cats = CATEGORICAL + (["bmi_class"] if include_bmi_class else [])
    num = NUMERIC + ["age_smoker_interaction"]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
        ]
    )