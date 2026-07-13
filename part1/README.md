# Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis 

# Prerequisites: 
I have uploaded the raw uncleaned used cars dataset (Cardetails_dataset.csv) in the part1 repository as referenced in the notebook code. 
Link for raw dataset is given below
https://github.com/552radhika-Dev/Final_Capstoneproj_Radhika/blob/main/part1/Cardetails_dataset.csv

# Deliverable:
The notebook produces cleaned_data.csv file, which will serve as the foundation for the next part.i.e.,Part2.

# Data Preparation & Exploration

Note : I have uploaded the dataset to my part1 folder in Final_Capstoneproj_Radhika repository. I have added this path to my code. 

**Project Stage:** Part 1 Deliverable
## 1. Project Overview & Objective
This project focuses on building an end-to-end machine learning pipeline to predict used car prices using historical listings data. Part 1 establishes clean, reliable data structures by resolving messy string formats, eliminating duplicates, downcasting categories to reduce memory, and analyzing underlying statistical skewness and outliers.

## 2. Feature Selection & Data Cleanup
- **Dropped Columns:** The `torque` column was intentionally dropped from the dataset. It contained unstructured text arrays (e.g., `"190Nm@ 2000rpm"`) requiring complex regex parsing, while its underlying predictive value was already captured cleaner by the `engine` (CC) and `max_power` (bhp) features.
- **Data Transformations:** Removed textual units (`kmpl`, `CC`, `bhp`) from `mileage`, `engine`, and `max_power` and converted them into clean, continuous floating-point variables using `pd.to_numeric()`.

## 3. Duplicate Detection & Data Quality
- **Identification Method:** Checked complete row duplications across all combined attributes via `df.duplicated().sum()`.
- **Rows Removed:** A total of **1,202 duplicate rows** were dropped using `df.drop_duplicates()`, scaling the unique dataset down to **6,926 records**.
- **Null Percentage Impact:** Dropping duplicate records updated both the missing cell count and total row count dynamically, keeping column null percentages uniform without distorting structural integrity.

## 4. Optimization & Memory Usage
- **Categorical Downcasting:** High-repetition string columns (`fuel`, `transmission`, and `owner`) were optimized into the `category` data type via `.astype('category')`.
- **Memory Reduction Report:**
  - **Before Optimization:** ~4.05 MB
  - **After Optimization:** ~1.43 MB
  - **Total Processing Efficiency Gain:** **64.71% memory savings**, allowing quicker matrix calculations during training.
  
## 5. Statistical Analysis & Skewness Profiling
A programmatic distribution scan using `df[col].skew()` returned the following insights for our core metrics:
- `km_driven`: 11.1709 (Extreme Positive Skew)
- `selling_price`: 4.1935 (Severe Positive Skew - Target Variable)
- `max_power`: 1.6596 (Moderate Positive Skew)
- `engine`: 1.1753 (Moderate Positive Skew)
- `mileage`: -0.1732 (Slight Negative Skew)
### Mean vs. Median Imputation Risks:
Because columns like our target `selling_price` and operational `km_driven` contain massive right-skewed profiles, a mathematical **Mean** calculation is heavily pulled upward by extreme outliers (luxury sports cars/commercial vehicles). If we used a biased Mean to fill missing entries, it would artificially overestimate values for standard consumer cars. Therefore, the **Median** is mathematically required as our robust center point.

## 6. Outlier Assessment Strategy (IQR Method)
Using the Interquartile Range ($IQR = Q3 - Q1$) threshold with lower/upper bounds ($Q1 - 1.5 \times IQR$ and $Q3 + 1.5 \times IQR$), outliers were profiled across our two primary metrics:
- **`selling_price` Outliers Detected:** 
- **`km_driven` Outliers Detected:** 
### Handling Plan for Part 2:
Outliers will **NOT** be dropped. High-value luxury cars and ultra-high-mileage vehicles represent valid, real-world marketplace transactions. Instead of dropping or capping, we will handle them in Part 2 by applying a mathematical **Log Transformation (`np.log1p`)** to the skewed distributions. This stabilizes structural variance and maps features into a normal shape that machine learning estimators can easily learn without numeric instability.
### Histogram Distribution Analysis
- **Feature Visualized:** `selling_price` (Identified as a highly skewed numeric target column).
- **Configuration:** Generated using `sns.histplot()` split into exactly 20 data bins.
- **Shape Description:** The distribution displays a severe **right-skewed (positive skew)** profile. The tall, massive tower on the far left shows that the vast majority of used cars cluster tightly within lower, budget-friendly consumer price brackets. As we move right across the X-axis, the graph rapidly thins out into a long, flat tail, indicating a tiny handful of ultra-premium luxury vehicles or exotic sports cars with exceptionally high price tags stretching out the scale.
### Scatter Plot Correlation Analysis
- **Features Visualized:** `max_power` (Independent Variable on X-Axis) vs. `selling_price` (Target Variable on Y-Axis).
- **Configuration:** Generated using `sns.scatterplot()`.
- **Direction & Strength Interpretation:** - **Direction:** The plot exhibits a clear **positive relationship** (as you move from left to right, the points trend upwards). This confirms that vehicles engineered with higher engine horsepower consistently command higher resale market values.
- **Strength:** The relationship displays a **strong, exponential strength profile**. In the lower horsepower brackets (under 100 bhp), the prices cluster tightly at a budget level. Once engine power crosses 150+ bhp, the pricing data points surge upwards rapidly. This demonstrates that `max_power` is a critical feature with a strong non-linear correlation to our target variable.
- ### Correlation Matrix Heatmap Analysis
- **Execution Summary:** Computed the pairwise Pearson correlation coefficients using `df.corr()` on all isolated numeric columns, visualized via `sns.heatmap(annot=True)`.
- **Highest Absolute Correlation Pair:** The strongest observed relationship in this dataset occurs between **`selling_price`** and **`max_power`** (engine brake horsepower), exhibiting a strong positive correlation value.

#### Causal vs. Non-Causal Relationship Breakdown
* While it is highly tempting to say a higher `max_power` *directly causes* a car to cost more, correlation alone does not strictly imply pure, isolated causation. 
* The Real Explanation: The strong correlation between high power and a high price tag is heavily influenced by a common hidden factor: **Brand Tier and Vehicle Segment (Luxury Classification)**. 
* **Plausible Alternative Explanations:**
    1.  **Luxury / Premium Segment Stature:** Exotic or premium luxury manufacturers (e.g., BMW, Mercedes-Benz, Audi) systematically build cars with larger, higher-horsepower engines (`max_power`). However, their high `selling_price` isn't purely determined by the raw engine power alone—it is driven by premium build quality, imported materials, advanced cabin tech, and brand equity markup. 
    2.  **Structural Inflation by Production Cost:** Higher-power engines inherently require more expensive engineering parts (strengthened transmissions, robust cooling systems, bigger brakes), which raises the base manufacturing cost, structurally forcing a higher retail and resale price.
### Missing Value Imputation Strategy
- **Target Columns:** `selling_price` and `km_driven`.
- **Chosen Statistic for Imputation:** **Median**

#### Statistical Justification
Both `selling_price` and `km_driven` are heavily **positively skewed (right-skewed)** variables. 
- In a positively skewed distribution, the column **mean** is significantly pulled upward and distorted by a tiny handful of extremely large outliers (e.g., ultra-luxury sports cars or rare million-kilometer fleet vehicles). 
- Using an artificially inflated mean to fill in missing values would systematically bias our entire dataset upwards. 
- Therefore, the **median** was chosen as the most robust and representative metric of central tendency because it is completely unaffected by extreme tail values. Missing records were successfully handled using `.fillna()`, and final verification via `.isnull().sum()` confirms zero missing entries remain.

### Spearman Rank vs. Pearson Correlation Analysis
- **Execution Summary:** Generated the rank-based Spearman correlation matrix (`df.corr(method='spearman')`) and compared it against the linear Pearson matrix to diagnose non-linear monotonic trends.

#### Top 3 Discrepancy Pairs Breakdown

- **(a) Relationship Classification:** [If |Spearman| > |Pearson|: **Monotonic but Non-Linear**. The variables move together consistently but not at a fixed proportional rate, causing Pearson's linear metric to underestimate the true relationship strength. / If |Pearson| >= |Spearman|: **Approximately Linear**.]
   - **(b) Feature-Selection Guidance Selection:** We will rely on the **[Spearman / Pearson]** coefficient because it captures the true underlying [non-linear / linear] dynamics without being skewed or suppressed by scaling transformations or outliers.

- **(a) Relationship Classification:** [If |Spearman| > |Pearson|: **Monotonic but Non-Linear**. The values move sequentially in step together but scale exponentially or logarithmically rather than along a straight line. / If |Pearson| >= |Spearman|: **Approximately Linear**.]
   - **(b) Feature-Selection Guidance Selection:** We will rely on the **[Spearman / Pearson]** coefficient for Part 2 feature prioritization, ensuring our models do not prematurely discard highly predictable non-linear features.

- **(a) Relationship Classification:** [If |Spearman| > |Pearson|: **Monotonic but Non-Linear**. / If |Pearson| >= |Spearman|: **Approximately Linear**.]
- **(b) Feature-Selection Guidance Selection:** We will utilize the **[Spearman / Pearson]** index because it provides a more accurate reflection of feature importance when feeding algorithms capable of mapping complex relationships.

### Grouped Aggregation Analysis
- **Execution Summary:** Grouped the cleaned dataset by the categorical feature `transmission` to inspect summary statistics (`mean`, `std`, `count`) against the target numeric variable `selling_price`.

#### Analysis Metrics & Findings
* **(a) Group Breakdown:** - The group with the **highest mean** selling price is **Automatic**.
  - The group with the **highest standard deviation (spread)** is **Automatic**.
* **(b) Predictive Modeling Variance Assessment:** - Yes, a high within-group standard deviation is a notable concern for a predictive model utilizing this categorical feature. High within-group variance means that even though a car belongs to the "Automatic" classification, its actual values span a massive range (e.g., from entry-level automatic hatchbacks to luxury supercars). Therefore, this single categorical feature *alone* is structurally insufficient to predict the exact price reliably and will require helper interaction columns (like `max_power` or `brand`) in Part 2 modeling.
* **(c) Mean Ratio & Predictive Signal:** - **Computed Ratio:** 
  - **Interpretation:** This ratio is significantly large (well above 1.5–2.0), which mathematically signals that the `transmission` feature carries a **strong predictive signal**. The huge disparity between the average price of an automatic car compared to a manual car guarantees that encoding this feature will provide the machine learning algorithms with high-value predictive boundaries during model development.
