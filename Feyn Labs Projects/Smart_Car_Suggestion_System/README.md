# Smart Car Suggestion System: Personalized Car Recommendations Based on Price Range

An interactive car recommendation mini-project that suggests cars from the Indian market matching a user's preferred number of doors, fuel type, and price range — backed by a Random Forest Regression model trained to estimate a car's ex-showroom price from its technical specifications.

##  Overview

This project combines a regression model with a simple interactive filter to recommend cars. A `RandomForestRegressor` is trained to predict a car's ex-showroom price from specs such as engine displacement, cylinders, drivetrain, and body dimensions. The user then enters their preferred number of doors, fuel type, and price range; the system filters the catalog for matching cars and shows each one alongside its model-predicted price.

##  Objectives

- Clean and preprocess a real-world Indian car specifications dataset
- Train a regression model to estimate ex-showroom price from vehicle specs
- Build an interactive recommendation flow filtered by doors, fuel type, and price range
- Evaluate model accuracy on unseen data
- Explore the dataset visually (pricing, dimensions, fuel type mix, luxury vs. budget segments)

##  Dataset

**File:** `cars_ds_final_2021.csv` — a detailed specifications dataset covering car models and variants sold in the Indian market, with pricing in Indian Rupees (Ex-Showroom Price).


**Key columns used in this project:**

| Column(s) | Description |
|---|---|
| `Make`, `Model`, `Variant` | Car identification |
| `Ex-Showroom_Price` | Listed price (cleaned from a "Rs. 1,23,456" string format to a float) |
| `Displacement`, `Cylinders`, `Valves_Per_Cylinder` | Engine specifications |
| `Drivetrain`, `Cylinder_Configuration`, `Emission_Norm`, `Fuel_Type`, `Body_Type` | Categorical specs (label-encoded for modeling) |
| `Height`, `Length`, `Width` | Body dimensions (cleaned from "…mm" string format to floats) |
| `Doors`, `Seating_Capacity` | Used to filter recommendations |
| `Type` | Vehicle type classification, used in exploratory visualizations |

##  Tech Stack

| Category | Tools/Libraries |
|---|---|
| Language | Python |
| Data Manipulation | Pandas |
| Modeling | Scikit-Learn (`RandomForestRegressor`, `train_test_split`, `LabelEncoder`) |
| Evaluation | Scikit-Learn (`mean_absolute_error`, `mean_squared_error`, `r2_score`) |
| Visualization | Matplotlib, Seaborn |

**`requirements.txt`** (adjust versions as needed):
```
pandas
scikit-learn
matplotlib
seaborn
```


##  Methodology

1. **Data Preprocessing**
   - Dropped the irrelevant `Index` column
   - Forward-filled missing values (`ffill`)
   - Cleaned `Ex-Showroom_Price` (removed "Rs." prefix and commas → float)
   - Cleaned `Height`, `Length`, `Width` (removed " mm" suffix → float)
   - Cleaned `Displacement` (removed " cc" suffix → float)
   - Label-encoded categorical columns: `Drivetrain`, `Emission_Norm`, `Cylinder_Configuration`, `Fuel_Type`, `Body_Type`

2. **Feature & Target Selection**
   - **Features:** `Displacement`, `Cylinders`, `Valves_Per_Cylinder`, `Drivetrain_encoded`, `Cylinder_Configuration_encoded`, `Emission_Norm_encoded`, `Fuel_Type_encoded`, `Height`, `Length`, `Width`, `Body_Type_encoded`
   - **Target:** `Ex-Showroom_Price`

3. **Model Training**
   - 80/20 train-test split
   - `RandomForestRegressor` with 100 estimators

4. **Personalized Recommendation Flow**
   - Prompts the user for number of doors, preferred fuel type, and min/max price range
   - Filters the dataset for cars matching all criteria
   - Runs the trained model on the filtered cars' specs to attach a `Predicted_Price`
   - Displays matching cars with their predicted price (or a "no matching cars found" message)

5. **Model Evaluation**
   - Computed MAE, MSE, and R² on the held-out test set

6. **Exploratory Visualizations**
   - Scatter plot: Displacement vs. Ex-Showroom Price
   - Histograms: distributions of Displacement, Height, Length, and Width
   - Pie chart: proportion of cars by Fuel Type
   - Bar charts: top luxury (highest-priced) and top budget (lowest-priced) cars
   - Bar chart: distribution of 5-seater cars across price bands (0–5L, 5–10L, 10–15L, 15–20L, 20L+ in Rupees)
   - Count plot: Body Type distribution split by vehicle Type

##  Results

### Model Performance 

| Metric | Value |
|---|---|
| Mean Absolute Error (MAE) | ₹9,59,178.13 |
| Mean Squared Error (MSE) | ~1.59 × 10¹³ |
| R² Score | 0.841 |

The model explains roughly **84% of the variance** in ex-showroom price using only technical specifications (engine, dimensions, drivetrain, etc.).


