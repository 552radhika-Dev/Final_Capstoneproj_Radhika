# Part 2: Supervised Machine Learning Model

# Task 1: Label and Feature Matrix
Feature Matrix (X): Consists of all independent features from `cleaned_data.csv` used to predict car values, with the continuous target column and the nominal column excluded. 
Regression Label (y_reg):Defined as the continuous numeric variable `selling_price` representing the raw transaction cost of the used car.
Classification Label (y_clf):A binary column derived by binarizing `y_reg` at its median value . Cars with a price strictly greater than the median are classified as `1` (Premium Price Tier), while cars below or equal to the median are classified as '0'(Standard Price Tier).

# Task 2: Categorical Feature Encoding Strategy
# Label Encoding Justification: 
The column 'owner' has a natural, step-by-step sequence based on vehicle wear and market depreciation. Categories are mapped directly to sequential integers to preserve this order: Test Drive Car as 0 (closest to brand new), followed by First Owner as 1, Second Owner as 2, Third Owner as 3, and Fourth & Above Owner as 4 (highest risk of wear). This tells the machine learning model that a higher ownership count introduces a progressive price penalty. 
# Onehot encoding : 
Avoiding False-Ordinal Relationships via One-Hot Encoding: The attributes for fuel type, seller type, and transmission type represent nominal categories with no natural ranking or inherent mathematical hierarchy. One-Hot encoding completely avoids this error by splitting nominal categories into isolated binary columns containing only 0 or 1. Furthermore, dropping the first dummy column establishes a clean baseline state. This eliminates perfect correlation between columns, preventing multicollinearity and protecting the underlying regression models from numerical instability.

# Task 3: Train-Test split and scaling
# Data Splitting Strategy
The cleaned and encoded dataset is split into separate training (80%) and testing (20%) subsets using train_test_split. 80% Training Set is used by the machine learning algorithms to learn the underlying patterns between features and the targets. 20% Test Set is Kept entirely hidden during training and used exclusively at the very end to evaluate how accurately the models perform on unseen, real-world data. By setting `test_size=0.2`, Python automatically allocates the remaining 0.8 (80%) to the training partition, ensuring a reliable and standard structural balance for model validation. By this way, Data leakage can be avoided.

# Task 4: Regression model — Linear Regression
Model 	Train MSE	Test MSE	Train R2	Test R2
Linear Regression	1,04,32,86,15,466.46	94,62,72,94,416.17	0.631	0.5685
Ridge Regression	1,04,32,86,20,702.91	94,62,52,46,603.07	0.631	0.5686
Ordinary Least Squares (OLS) focuses entirely on minimizing training errors, which can cause its mathematical coefficients to grow excessively large or become highly sensitive to noise or correlated features. Ridge Regression modifies this behavior by adding an regularization penalty to the cost function, mathematically calculated as the sum of squared coefficients. This extra penalty forces the model to shrink the size of its coefficients toward zero, reducing overall model complexity. The `alpha` parameter acts as a control dial for this penalty: setting `alpha=0` yields identical results to standard OLS, while increasing it applies a heavier penalty that lowers model variance and stabilizes predictions on unseen test data.

# Task 5a: Classification model — Logistic Regression
# Handling Class Imbalance
To address class imbalances where one class falls beneath the 35% threshold, the hyperparameter 'class_weight'='balanced' was specified within the Logistic Regression constructor. This built-in approach automatically scales the model's loss function inversely proportional to class frequencies in the training data. This choice was favored over SMOTE because it natively penalizes misclassifications of the minority class during optimization without synthetically generating artificial datapoints, keeping our pipeline clean, fast, and leak-free.
# Confusion matrix
[[606  82]
 [ 97 601]]
# The First Row (Actual Cheap Cars - Class 0)
606 (True Negatives): There were 606 cheap cars that your model correctly predicted as cheap.
82 (False Positives): There were 82 cheap cars that your model accidentally flagged as expensive.
# The Second Row (Actual Expensive Cars - Class 1)
97 (False Negatives): There were 97 expensive cars that your model completely missed and accidentally labeled as cheap.
601 (True Positives): There were 601 expensive cars that your model correctly predicted as expensive.
# precision = TP/(TP+FP)  |   Recall = TP/(TP+FN)
For this specific classification framework—where the positive class (`1`) represents a vehicle valued **above the median price** (a premium/expensive vehicle)—**Precision is the more important metric**.
# The Business Impact:
A False Positive means the model misclassifies a standard, low-budget car as a premium luxury vehicle. If a dealership or pricing tool utilizes these predictions to set automated sticker prices, they will significantly overprice average cars. This directly drives away potential buyers, ruins customer trust, and leaves inventory rotting on the lot.
# The Contrast with Recall:
A False Negative means an expensive car is accidentally labeled as cheap. While this leads to missed profit margins by underpricing, it ensures the vehicle sells rapidly. Overpricing due to low Precision is a much more structural liability to the business model than underpricing. Therefore, maximizing Precision is our primary objective here.
# Interpretation of the Area Under the ROC Curve (AUC)
The Area Under the ROC Curve (AUC) serves as a threshold-independent metric tracking the model's absolute capability to separate the two pricing classes (cheap vs. expensive). 
An AUC value measures the probability that the model will correctly assign a higher classification probability to a randomly chosen actual positive instance (expensive car) than a randomly chosen actual negative instance (cheap car). 
An AUC of `0.50` indicates performance no better than an uninformative coin toss.
Our baseline Logistic Regression model achieves an AUC of 0.9460, demonstrating an excellent and mathematically robust capability to distinguish luxury high-tier vehicles from standard budget-tier vehicles across all baseline operational limits.

# Task 5b: Decision-threshold sensitivity
=== Decision-Threshold Table ===
Threshold Precision Recall     F1
     0.30    0.8102 0.9298 0.8659
     0.40    0.8419 0.8926 0.8665
     0.50    0.8799 0.8610 0.8704
     0.60    0.8964 0.8181 0.8554
     0.70    0.9216 0.7751 0.8420
# Maximizing the F1-Score:
Based on the experimental grid, the threshold that maximizes the F1-Score on this dataset is 0.50. This represents the best mathematical compromise between Precision and Recall.
# Strategic Metric Importance:
For this specific application (identifying premium cars priced above the median value), **Precision** remains the more important metric. A False Positive means the system flags a standard budget car as a high-tier asset. If an automated pricing tool relies on this, it will overprice a cheap vehicle—pushing away buyers and leaving stale inventory on the lot.
# Operational Adjustments & Trade-offs:
To optimize for Precision, we must raise the decision threshold. By forcing the model to be more conservative, it only flags vehicles as expensive when it possesses high mathematical certainty, minimizing False Positive errors. 
# The Cost of Doing So:
Raising the threshold introduces a severe penalty to **Recall**. The model becomes overly strict, which means it will experience a high volume of False Negatives —missing many actual expensive cars and mislabeling them as cheap, causing the business to lose out on higher profit margins by underpricing them. 

# Task 6: Regularization experiment on Logistic Regression
=== Logistic Regression Regularization Comparison ===
          Model Hyperparameter Precision Recall ROC AUC
              Baseline (C=1.0)    0.8799 0.8610  0.9460
Strong Regularization (C=0.01)    0.8723 0.8711  0.9468
Reducing 'C' to 0.01 provided a slight improvement in overall model generalization, as shown by the increase in both Recall (from 0.8610 to 0.8711) and ROC AUC (from 0.9460 to 0.9468). Although Precision decreased marginally from 0.8799 to 0.8723, the model became more balanced and robust. This indicates that the stronger regularization successfully penalized minor noise in the features, helping the model separate the price classes more effectively on unseen test data.

# Task 7: Bootstrap confidence interval for AUC difference
To quantify the stability of the performance gap (AUC_C=1.0 - AUC_C=0.01), a bootstrapping procedure is executed over 500 resampled iterations of the test set:
# Mean AUC Difference: -0.000857
# 95% Confidence Interval (CI): [-0.003055, 0.001309]
# Reliability Analysis:
Because the 95% confidence interval explicitly includes zero (spanning from a negative value to a positive value), the performance difference between the baseline model and the strongly regularized model is not statistically reliable. This indicates that the tiny operational advantage observed on our single test split is likely due to random data sampling noise rather than a true structural superiority of one model configuration over the other.




