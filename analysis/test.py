#Classical association rule mining algorithms (Apriori, Eclat, FP-Growth) 
# were used to discover frequent product combinations. However, these methods 
# focus solely on frequency and ignore economic value. To address this 
# limitation, we extend the analysis with a utility-aware approach inspired 
# by UP-Tree, incorporating product price and quantity to highlight 
# high-revenue itemsets. Finally, gradient boosting models (XGBoost / LightGBM) 
# are used to predict reorder likelihood, enabling personalized recommendations.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
# Get warnings distinct from errors
import warnings
# Ignore them to avoid slowdown
warnings.filterwarnings('ignore')

DATA_PATH = Path(__file__).parent.parent / "data"

print("   Loading orders.csv (sampling for analysis)...")
orders = pd.read_csv(DATA_PATH / "orders.csv")  

print("   Loading order_products__train.csv...")
order_products_train = pd.read_csv(DATA_PATH / "order_products__train.csv")

print("   Loading order_products__prior.csv (all)...")
order_products_prior = pd.read_csv(DATA_PATH / "order_products__prior.csv")

datasets = {
    "orders": orders,
    "order_products_train": order_products_train,
    "order_products_prior": order_products_prior
}

for name, df in datasets.items():
    print(f"\n {name.upper()}")
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

