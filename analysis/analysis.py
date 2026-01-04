# =============================================================================
#
# Goal : Instacart Data Analysis
# Group 11 (Data Genes)
# Date Started : 29/12/2025
# Date Completion : 04/01/2026
#
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
# Get warnings distinct from errors
import warnings
# Ignore them to avoid slowdown
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# =============================================================================
# DATA LOADING
# =============================================================================
print("=" * 60)
print("INSTACART MARKET BASKET ANALYSIS")
print("Group 11 - Data Genes")
print("=" * 60)

# Define data path
DATA_PATH = Path(__file__).parent.parent / "data"

print("\n Loading datasets...")

# Load smaller files first
aisles = pd.read_csv(DATA_PATH / "aisles.csv")
departments = pd.read_csv(DATA_PATH / "departments.csv")
products = pd.read_csv(DATA_PATH / "products.csv")

# Load order data (sampling for large files due to memory constraints)
print("   Loading orders.csv (sampling for analysis)...")
orders = pd.read_csv(DATA_PATH / "orders.csv", nrows=500000)  # Sample for analysis

print("   Loading order_products__train.csv...")
order_products_train = pd.read_csv(DATA_PATH / "order_products__train.csv")

print("   Loading order_products__prior.csv (sampling for analysis)...")
order_products_prior = pd.read_csv(DATA_PATH / "order_products__prior.csv", nrows=1000000)

print(" All datasets loaded successfully!\n")

# =============================================================================
# DATA OVERVIEW
# =============================================================================
print("=" * 60)
print("1. DATA OVERVIEW")
print("=" * 60)

datasets = {
    "aisles": aisles,
    "departments": departments,
    "products": products,
    "orders": orders,
    "order_products_train": order_products_train,
    "order_products_prior": order_products_prior
}

for name, df in datasets.items():
    print(f"\n {name.upper()}")
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# =============================================================================
# DATA QUALITY CHECK
# =============================================================================
print("\n" + "=" * 60)
print("2. DATA QUALITY CHECK")
print("=" * 60)

for name, df in datasets.items():
    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    print(f"\n {name.upper()}")
    print(f"   Missing values: {missing:,}")
    print(f"   Duplicate rows: {duplicates:,}")

# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "=" * 60)
print("3. DESCRIPTIVE STATISTICS")
print("=" * 60)

# Orders statistics
print("\n ORDERS STATISTICS:")
print(f"   Total orders (sample): {orders['order_id'].nunique():,}")
print(f"   Total users (sample): {orders['user_id'].nunique():,}")
print(f"   Order number range: {orders['order_number'].min()} - {orders['order_number'].max()}")

# Order day of week distribution
print("\n Orders by Day of Week:")
dow_mapping = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 
               4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
orders_dow = orders['order_dow'].value_counts().sort_index()
for day, count in orders_dow.items():
    print(f"   {dow_mapping.get(day, day)}: {count:,} orders ({count/len(orders)*100:.1f}%)")

# Order hour distribution
print("\n Peak Shopping Hours:")
orders_hour = orders['order_hour_of_day'].value_counts().sort_values(ascending=False).head(5)
for hour, count in orders_hour.items():
    print(f"   {hour:02d}:00 - {count:,} orders")

# Products statistics
print("\n PRODUCTS STATISTICS:")
print(f"   Total products: {products['product_id'].nunique():,}")
print(f"   Total aisles: {aisles['aisle_id'].nunique():,}")
print(f"   Total departments: {departments['department_id'].nunique():,}")

# Reorder statistics
print("\n REORDER STATISTICS:")
reorder_rate_train = order_products_train['reordered'].mean() * 100
reorder_rate_prior = order_products_prior['reordered'].mean() * 100
print(f"   Reorder rate (train): {reorder_rate_train:.1f}%")
print(f"   Reorder rate (prior): {reorder_rate_prior:.1f}%")

# =============================================================================
# TOP PRODUCTS ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("4. TOP PRODUCTS ANALYSIS")
print("=" * 60)

# Merge to get product names
order_products_merged = order_products_train.merge(products, on='product_id')
order_products_merged = order_products_merged.merge(aisles, on='aisle_id')
order_products_merged = order_products_merged.merge(departments, on='department_id')

# Top 10 products
print("\n Top 10 Most Ordered Products:")
top_products = order_products_merged['product_name'].value_counts().head(10)
for i, (product, count) in enumerate(top_products.items(), 1):
    print(f"   {i:2d}. {product}: {count:,} orders")

# Top aisles
print("\n Top 10 Most Popular Aisles:")
top_aisles = order_products_merged['aisle'].value_counts().head(10)
for i, (aisle, count) in enumerate(top_aisles.items(), 1):
    print(f"   {i:2d}. {aisle}: {count:,} orders")

# Top departments
print("\n Top Departments:")
top_departments = order_products_merged['department'].value_counts()
for i, (dept, count) in enumerate(top_departments.items(), 1):
    print(f"   {i:2d}. {dept}: {count:,} orders ({count/len(order_products_merged)*100:.1f}%)")

# =============================================================================
# CART SIZE ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("5. CART SIZE ANALYSIS")
print("=" * 60)

cart_sizes = order_products_train.groupby('order_id').size()
print(f"\n Cart Size Statistics:")
print(f"   Mean items per order: {cart_sizes.mean():.1f}")
print(f"   Median items per order: {cart_sizes.median():.1f}")
print(f"   Min items: {cart_sizes.min()}")
print(f"   Max items: {cart_sizes.max()}")
print(f"   Std deviation: {cart_sizes.std():.1f}")

# =============================================================================
# CUSTOMER BEHAVIOR ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("6. CUSTOMER BEHAVIOR ANALYSIS")
print("=" * 60)

# Days since prior order
days_since_prior = orders['days_since_prior_order'].dropna()
print(f"\n Days Between Orders:")
print(f"   Mean: {days_since_prior.mean():.1f} days")
print(f"   Median: {days_since_prior.median():.1f} days")
print(f"   Most common: {days_since_prior.mode().values[0]:.0f} days")

# Orders per user
orders_per_user = orders.groupby('user_id').size()
print(f"\n Orders per User:")
print(f"   Mean: {orders_per_user.mean():.1f} orders")
print(f"   Median: {orders_per_user.median():.1f} orders")
print(f"   Max: {orders_per_user.max()} orders")

# =============================================================================
# DATA VISUALIZATIONS (saved to files)
# =============================================================================
print("\n" + "=" * 60)
print("7. GENERATING VISUALIZATIONS")
print("=" * 60)

# Create figures directory
FIGURES_PATH = Path(__file__).parent / "figures"
FIGURES_PATH.mkdir(exist_ok=True)

# Figure 1: Orders by Day of Week
fig, ax = plt.subplots(figsize=(10, 6))
orders_dow_plot = orders['order_dow'].value_counts().sort_index()
bars = ax.bar([dow_mapping[i] for i in orders_dow_plot.index], orders_dow_plot.values, color='steelblue')
ax.set_xlabel('Day of Week', fontsize=12)
ax.set_ylabel('Number of Orders', fontsize=12)
ax.set_title('Orders Distribution by Day of Week', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIGURES_PATH / 'orders_by_day.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: orders_by_day.png")

# Figure 2: Orders by Hour
fig, ax = plt.subplots(figsize=(12, 6))
orders_hour_plot = orders['order_hour_of_day'].value_counts().sort_index()
ax.plot(orders_hour_plot.index, orders_hour_plot.values, marker='o', linewidth=2, markersize=8, color='coral')
ax.fill_between(orders_hour_plot.index, orders_hour_plot.values, alpha=0.3, color='coral')
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Number of Orders', fontsize=12)
ax.set_title('Orders Distribution by Hour of Day', fontsize=14, fontweight='bold')
ax.set_xticks(range(0, 24))
plt.tight_layout()
plt.savefig(FIGURES_PATH / 'orders_by_hour.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: orders_by_hour.png")

# Figure 3: Top 15 Products
fig, ax = plt.subplots(figsize=(12, 8))
top_15_products = order_products_merged['product_name'].value_counts().head(15)
ax.barh(range(len(top_15_products)), top_15_products.values, color='seagreen')
ax.set_yticks(range(len(top_15_products)))
ax.set_yticklabels(top_15_products.index)
ax.set_xlabel('Number of Orders', fontsize=12)
ax.set_title('Top 15 Most Ordered Products', fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(FIGURES_PATH / 'top_products.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: top_products.png")

# Figure 4: Top Departments Pie Chart
fig, ax = plt.subplots(figsize=(10, 10))
top_10_depts = order_products_merged['department'].value_counts().head(10)
colors = plt.cm.Set3(np.linspace(0, 1, len(top_10_depts)))
wedges, texts, autotexts = ax.pie(top_10_depts.values, labels=top_10_depts.index, 
                                   autopct='%1.1f%%', colors=colors, startangle=90)
ax.set_title('Orders Distribution by Department (Top 10)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_PATH / 'departments_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: departments_distribution.png")

# Figure 5: Cart Size Distribution
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(cart_sizes, bins=50, color='purple', alpha=0.7, edgecolor='black')
ax.axvline(cart_sizes.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {cart_sizes.mean():.1f}')
ax.axvline(cart_sizes.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {cart_sizes.median():.1f}')
ax.set_xlabel('Number of Items in Cart', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Cart Sizes', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_PATH / 'cart_size_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: cart_size_distribution.png")

# Figure 6: Reorder Rate by Department
dept_reorder = order_products_merged.groupby('department')['reordered'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(range(len(dept_reorder)), dept_reorder.values * 100, color='teal')
ax.set_yticks(range(len(dept_reorder)))
ax.set_yticklabels(dept_reorder.index)
ax.set_xlabel('Reorder Rate (%)', fontsize=12)
ax.set_title('Reorder Rate by Department', fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(FIGURES_PATH / 'reorder_rate_by_department.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: reorder_rate_by_department.png")

# Figure 7: Days Since Prior Order Distribution
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(days_since_prior, bins=31, color='dodgerblue', alpha=0.7, edgecolor='black')
ax.axvline(days_since_prior.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {days_since_prior.mean():.1f} days')
ax.set_xlabel('Days Since Prior Order', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Days Between Orders', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_PATH / 'days_since_prior_order.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: days_since_prior_order.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("8. ANALYSIS SUMMARY")
print("=" * 60)
print("""
KEY FINDINGS:
================

1. DATASET OVERVIEW:
   - Large-scale e-commerce dataset from Instacart
   - Contains orders, products, aisles, and departments data
   - Over 3 million orders from 200k+ users

2. SHOPPING PATTERNS:
   - Peak shopping days: Weekends (Saturday & Sunday)
   - Peak shopping hours: 10 AM - 4 PM
   - Average time between orders: ~11 days
   - Weekly shopping pattern is common (7 days, 30 days)

3. PRODUCT INSIGHTS:
   - Top products: Bananas, Organic items, Fresh fruits
   - Most popular aisles: Fresh fruits, Fresh vegetables
   - Dominant departments: Produce, Dairy & Eggs, Snacks

4. CUSTOMER BEHAVIOR:
   - Average cart size: ~10 items per order
   - High reorder rate (~59%) - customers have habitual purchases
   - Produce has highest reorder rate - fresh items bought regularly

5. ML POTENTIAL:
   - Predict reorders (binary classification)
   - Recommend products (recommendation system)
   - Customer segmentation (clustering)
   - Basket analysis (association rules)
""")

print("\n" + "=" * 60)
print(" ANALYSIS COMPLETE!")
print("=" * 60)
print(f"\n Visualizations saved to: {FIGURES_PATH}")

