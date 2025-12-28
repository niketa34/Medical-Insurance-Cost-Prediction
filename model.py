import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def get_models(seed:int=42):
    return {
        "linear": LinearRegression(),
        "lasso": Lasso(alpha=0.001, random_state=seed),
        "ridge": Ridge(alpha=1.0, random_state=seed),
        "rf": RandomForestRegressor(n_estimators=300, max_depth=None, random_state=seed),
        "xgb": XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            reg_lambda=1.0,
        ),
    }