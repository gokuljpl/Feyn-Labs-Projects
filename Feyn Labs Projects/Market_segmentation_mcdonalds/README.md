# McDonald's Customer Market Segmentation

An unsupervised machine learning project that segments McDonald's customers using demographic, psychographic, and behavioral survey data, combining PCA with K-Means clustering to uncover distinct customer segments.

##  Overview

This project analyzes a customer survey dataset capturing how respondents perceive McDonald's (e.g., tasty, cheap, greasy, healthy), along with their demographics (age, gender), overall sentiment ("Like" score), and visit frequency. Using dimensionality reduction (PCA) and K-Means clustering, respondents are grouped into distinct segments, which are then profiled and evaluated to identify a priority target segment for marketing.

##  Objectives

- Explore demographic (age, gender) and psychographic (perception, sentiment) patterns in the survey data
- Encode categorical perception attributes for clustering
- Reduce dimensionality with PCA and inspect feature loadings
- Determine the optimal number of clusters using the Elbow Method
- Segment respondents using K-Means and profile each cluster
- Evaluate clusters by sentiment and visit frequency to identify a target segment

##  Dataset

**File:** `mcdonalds.csv` — 1,453 survey respondents


| Column(s) | Type | Description |
|---|---|---|
| `yummy`, `convenient`, `spicy`, `fattening`, `greasy`, `fast`, `cheap`, `tasty`, `expensive`, `healthy`, `disgusting` | Binary (Yes/No) | Respondent's perception of McDonald's on each attribute |
| `Like` | Ordinal | Sentiment score from "I hate it! -5" to "I love it! +5" |
| `Age` | Numeric | Respondent age (18–71) |
| `VisitFrequency` | Categorical | Never, Once a year, Every three months, Once a month, Once a week, More than once a week |
| `Gender` | Categorical | Female / Male |

**Sample distributions from EDA:**
- Gender: 788 Female / 665 Male
- Visit frequency: Once a month (439) > Every three months (342) > Once a year (252) > Once a week (235) > Never (131) > More than once a week (54)
- Age is broadly spread from 18–71, with the highest concentration of respondents in their 30s–60s

##  Tech Stack

| Category | Tools/Libraries |
|---|---|
| Language | Python |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Categorical Encoding | Scikit-Learn (`LabelEncoder`) |
| Dimensionality Reduction | Scikit-Learn (`PCA`), `bioinfokit` (PCA biplot) |
| Cluster Selection | `yellowbrick` (`KElbowVisualizer`) |
| Clustering | Scikit-Learn (`KMeans`) |
| Segment Description | `statsmodels` (mosaic plots), `collections.Counter` |


**`requirements.txt`** (adjust versions as needed):
```
pandas
numpy
seaborn
matplotlib
scikit-learn
yellowbrick
bioinfokit
statsmodels
```


##  Methodology

1. **Exploratory Data Analysis**
   - Reviewed dataset shape, data types, and missing values
   - Examined value counts for `Like`, `Age`, `VisitFrequency`, and `Gender`
   - Plotted distributions/count plots for all features and reviewed feature correlations

2. **Demographic Segmentation**
   - Gender split visualized with a pie chart (more female than male respondents)
   - Age distribution visualized with a count plot

3. **Psychographic Segmentation**
   - Visualized `Like` (renamed for readability) against `Age` using a swarm plot to see how sentiment varies across age groups

4. **Data Preprocessing**
   - Dropped `Like`, `Age`, `VisitFrequency`, and `Gender`, isolating the 11 binary Yes/No perception attributes
   - Applied `LabelEncoder` to convert Yes/No responses into 1/0

5. **PCA**
   - Scaled the encoded attributes and applied PCA (11 components)
   - Reviewed explained variance ratios and component loadings (visualized as a heatmap)
   - Generated a 2D PCA biplot to inspect how attributes relate to the principal components

6. **Cluster Selection**
   - Used `yellowbrick`'s `KElbowVisualizer` (k = 1–12) to identify the optimal number of clusters

7. **K-Means Clustering**
   - Trained a final K-Means model with **k = 4** (`random_state=0`)
   - Visualized clusters in PC1/PC2 space with cluster centroids

8. **Describing Segments**
   - Cross-tabulated cluster assignment against `Like` and `Gender`, visualized with mosaic plots
   - Compared `Age` distribution across clusters with a boxplot

9. **Target Segment Selection**
   - Label-encoded `VisitFrequency`, `Like`, and `Gender`
   - Computed per-cluster means for each and merged into a single segment summary table
   - Plotted mean `VisitFrequency` vs. mean `Like` per cluster to visually evaluate which segment(s) to prioritize

