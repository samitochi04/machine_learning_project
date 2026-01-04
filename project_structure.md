# Machine Learning Project: GROUP 11

## Project Structure :

```
Group_Project/
├── analysis/
│   ├── analysis.py          # Data analysis script
│   └── figures/             # Generated visualizations
├── data/
│   ├── aisles.csv           # Aisle information (134 aisles)
│   ├── departments.csv      # Department information (21 departments)
│   ├── orders.csv           # Order metadata (~3.4M orders)
│   ├── order_products__prior.csv  # Prior order product details
│   ├── order_products__train.csv  # Training order product details
│   └── products.csv         # Product catalog (~50K products)
├── models/                  # ML models
|
└── notebooks/               # Jupyter notebooks
```
---

## Project Goals :

### Primary Objective:
Build a **Market Basket Analysis & Product Recommendation System** using Instacart e-commerce data.

### Specific Goals:

1. **Predictive Modeling** - Predict which products a user will reorder based on historical purchase patterns

2. **Product Recommendations** - Build a recommendation system to suggest products to customers

3. **Customer Segmentation** - Cluster customers based on purchasing behavior

4. **Market Basket Analysis** - Discover product associations and frequently bought together items

5. **Web Application** - Deploy an interactive dashboard/application for predictions and recommendations

---

## Short Data Analysis :

### Dataset Overview:
- **Source:** Instacart Market Basket Analysis (Kaggle competition)
- **Size:** ~680 MB total dataset
- **Records:** ~3.4 million orders from 200k+ users

### Data Files:
| File | Records | Description |
|------|---------|-------------|
| orders.csv | 3.4M | Order metadata (user, timing, eval set) |
| products.csv | 49,688 | Product catalog with aisle/department |
| aisles.csv | 134 | Aisle categories |
| departments.csv | 21 | Department categories |
| order_products__prior.csv | 32M | Products in prior orders |
| order_products__train.csv | 1.4M | Products in training orders |

### Key Insights:

**Shopping Patterns:**
- Peak days: Saturday & Sunday (weekend shopping)
- Peak hours: 10 AM - 4 PM
- Average days between orders: ~11 days
- Weekly patterns common (7-day, 30-day cycles)

**Product Trends:**
- Top products: Bananas, Organic items, Fresh fruits
- Top aisles: Fresh fruits, Fresh vegetables, Packaged cheese
- Top departments: Produce (30%), Dairy & Eggs (17%), Snacks (10%)

**Customer Behavior:**
- Average cart size: ~10 items per order
- Reorder rate: ~59% (high loyalty)
- Produce with highest reorder rate

---

### Tools :

- **Git & GitHub** : Version control & collaboration. Repo : https://github.com/samitochi04/machine_learning_project

- **Trello** : Kanban board for Project Management

- **Web Development & DevOps** : React, NodeJs, Python, Supabase, Docker, Coolify

- **Data Science** : 
  - Python (pandas, numpy, matplotlib, seaborn)
  - Scikit-learn (classification, clustering)
  - XGBoost / LightGBM (gradient boosting models)
  - MLxtend (association rules, frequent patterns)

- **Data Engineering** : 
  - APIs (FastAPI)
  - Supabase (database)
  - Data pipelines & ETL

- **Business Analytics** : 
  - Excel (quick analysis)
  - pandas (data manipulation)
  - matplotlib (data visualization)

---