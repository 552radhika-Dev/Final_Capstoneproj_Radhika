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
## Data Splitting Strategy
The cleaned and encoded dataset is split into separate training (80%) and testing (20%) subsets using train_test_split. 80% Training Set is used by the machine learning algorithms to learn the underlying patterns between features and the targets. 20% Test Set is Kept entirely hidden during training and used exclusively at the very end to evaluate how accurately the models perform on unseen, real-world data. By setting `test_size=0.2`, Python automatically allocates the remaining 0.8 (80%) to the training partition, ensuring a reliable and standard structural balance for model validation. By this way, Data leakage can be avoided.

# Task 4: Regression model — Linear Regression
Model 	Train MSE	Test MSE	Train R2	Test R2
Linear Regression	1,04,32,86,15,466.46	94,62,72,94,416.17	0.631	0.5685
Ridge Regression	1,04,32,86,20,702.91	94,62,52,46,603.07	0.631	0.5686
Ordinary Least Squares (OLS) focuses entirely on minimizing training errors, which can cause its mathematical coefficients to grow excessively large or become highly sensitive to noise or correlated features. Ridge Regression modifies this behavior by adding an regularization penalty to the cost function, mathematically calculated as the sum of squared coefficients. This extra penalty forces the model to shrink the size of its coefficients toward zero, reducing overall model complexity. The `alpha` parameter acts as a control dial for this penalty: setting `alpha=0` yields identical results to standard OLS, while increasing it applies a heavier penalty that lowers model variance and stabilizes predictions on unseen test data.
