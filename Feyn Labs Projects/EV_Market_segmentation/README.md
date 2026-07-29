# Market Segmentation of EVs in India

An unsupervised machine learning project that segments the Indian electric vehicle (EV) market using exploratory data analysis, PCA, and K-Means clustering.

##  Overview

This project explores the Indian EV landscape from two angles: **state-wise adoption trends** (how many 2-wheelers, 3-wheelers, buses, and passenger cars are registered across Indian states) and **vehicle-level specifications** (brand, body style, price, range, acceleration, etc.). It then applies K-Means clustering on the specification data to segment EVs into distinct market groups, helping identify natural product categories based on performance, price, and design characteristics.

##  Objectives

- Explore and visualize state-wise EV adoption across vehicle categories (2W, 3W, buses, passenger cars)
- Explore EV specifications across brands (body style, segment, seats, powertrain, plug type, price, performance)
- Engineer and scale features suitable for clustering
- Reduce dimensionality using PCA
- Determine the optimal number of clusters using the Elbow Method
- Segment the EV market using K-Means and profile each resulting cluster

##  Datasets

The project uses two datasets:

| Dataset | File | Description |
|---|---|---|
| Dataset 1 | `EV Stats-1.csv` | State-wise registration counts across vehicle categories — two-wheelers, three-wheelers, buses, and passenger cars |
| Dataset 2 | `ElectricCarData_Norm.csv` | EV specifications — Brand, Model, BodyStyle, Segment, Seats, PowerTrain, PlugType, Accel, TopSpeed, Range, Efficiency, FastCharge, RapidCharge, PriceEuro |


##  Tech Stack

| Category | Tools/Libraries |
|---|---|
| Language | Python |
| Data Manipulation | Pandas |
| Visualization | Matplotlib, Seaborn |
| Preprocessing | Scikit-Learn (`StandardScaler`) |
| Dimensionality Reduction | Scikit-Learn (`PCA`) |
| Clustering | Scikit-Learn (`KMeans`) |
| Model Selection Utilities | Scikit-Learn (`train_test_split`) |


**`requirements.txt`** (adjust versions as needed):
```
pandas
matplotlib
seaborn
scikit-learn
```


##  Methodology

1. **Exploratory Data Analysis**
   - Dataset shapes, info, and statistical summaries for both datasets
   - State-wise breakdowns of two-wheeler, three-wheeler, bus, and passenger car EV registrations
   - Brand, body style, seat count, segment, powertrain, and plug type distributions
   - Price distribution and price-vs-segment analysis
   - Correlation matrix across numeric specification features

2. **Data Cleaning & Feature Engineering**
   - Converted specification columns (`TopSpeed`, `Efficiency`, `FastCharge`, `Accel`, `Range`) from string/mixed formats to numeric types
   - Encoded categorical features: `PowerTrain` (Rear Wheel Drive / Front Wheel Drive / All Wheel Drive → 0/1/2) and `RapidCharge` (possible/not possible → 0/1)
   - Selected clustering features: `Accel`, `TopSpeed`, `Efficiency`, `FastCharge`, `Range`, `RapidCharge`, `Seats`, `PriceEuro`, `PowerTrain`

3. **Preprocessing for Clustering**
   - Standardized features using `StandardScaler`
   - Reduced dimensionality with PCA (9 components), reviewing explained variance ratios per component

4. **Cluster Selection & Modeling**
   - Used the **Elbow Method** (WCSS across k = 1–10) to select the optimal number of clusters
   - Trained a final **K-Means** model with **k = 4**
   - Visualized clusters in principal component space with cluster centroids

5. **Cluster Profiling**
   - Computed per-cluster mean values for acceleration, top speed, range, efficiency, seats, and price
   - Built descriptive profiles per cluster (brands, models, powertrain types, plug types, body styles, seat range, price range)

