# Part 3 — Advanced Modeling — Ensembles, Tuning, and Full ML Pipeline

# Pre-requisites:
I uploaded the scaled (X_train and X_test) , y_clf_train and y_clf_test files in NPY format in the part 3 repository which i got as a output of part 2.

# Deliverables:
The notebook produces a trained model file (best_model.pkl).
It generates a comprehensive evaluation report, including regression metrics (MSE, $R^2$), classification metrics (Confusion Matrix, Precision, Recall, F1-Score), an ROC/AUC plot, and a threshold sensitivity analysis table

# Task1: Decision Tree Baseline
# Overfitting Evaluation: 
Yes, the baseline Decision Tree shows severe signs of overfitting. It achieved a Training Accuracy of 100.00% but a significantly lower Testing Accuracy of 82.14%. This large gap proves the model has memorized the training data rather than generalizing well to unseen patterns.Why Decision Trees are High-Variance: Decision Trees are high-variance models because they make split choices greedily at each node, optimizing solely for immediate data isolation without looking ahead or revisiting past decisions. Without depth constraints, they continuously build complex, unique branches for every outlier and noise point in the training dataset, causing the final structure to change drastically given even minor changes in the input data.

# Task2: Controlled Decision Tree
# Role of `max_depth`:
Limits the maximum number of question levels the tree can grow. Restricting depth prevents the tree from creating overly complex rules for individual data points, which reduces variance (overfitting) at the cost of introducing a small amount of bias.
# Role of `min_samples_split`:
Dictates the minimum number of data samples required inside a node before it is legally allowed to split further. By requiring at least 20 samples to split, it stops the model from responding to random noise within tiny subsets of data.
# Train/Test Gap Comparison:
Unconstrained Tree showed severe overfitting with a massive gap between a 100.00% training accuracy and an 82.14% testing accuracy. 
# Controlled Tree:
Controlled tree Significantly shrinks this generalization gap. While training accuracy drops because the tree is no longer allowed to memorize every row, the test accuracy remains stable and robust, demonstrating a much healthier, more generalizable model.

# Task3: Gini vs Entropy comparison
The mathematical formulas used to evaluate data chaos are Gini Impurity (1 - summationp_i^2) and Entropy ($-\sum p_i \log_2(p_i)$). When a node achieves a value of exactly 0, it means it has reached absolute purity where every single sample belongs to only one class with 100% certainty. Here in this dataset, both mathematical criteria relied heavily on the same patterns, with certain acting as the primary splitters responsible for over 83% of the total impurity reduction. Because these two features successfully isolated your target categories into clean, low-impurity pockets right at the top of the trees, both models established highly stable decision boundaries that accurately generalized to unseen testing data, yielding test accuracies of 86.87% (Gini) and 87.09% (Entropy).

# Task4: Random Forest
Random Forest computes feature importance by measuring the average reduction in Gini impurity brought by a feature across all splits, averaged over all 100 trees in the ensemble. Unlike a linear regression coefficient—which calculates a simple directional weight assuming a strict linear layout—Random Forest importance measures a feature's pure predictive power and its ability to cleanly partition complex data structures. 

The model relies on bagging, where each individual tree is trained on a distinct bootstrap sample taken with replacement from your training data, while restricting each node split to evaluate a random subset of $\sqrt{\text{total features}}$. This double-randomization creates diverse, uncorrelated trees that make unique mistakes. When the ensemble averages out all these individual predictions, it completely breaks down the high variance of a single deep tree, creating stable decision boundaries that generalize perfectly to unseen data.

# Task4a: Gradient Boosting Analysis
Gradient Boosting is an ensemble technique that builds trees sequentially rather than independently, meaning each new tree is specifically trained to correct the precise errors (residuals) made by the previous trees. Controlled by a learning_rate of 0.1, each tree scales down its corrections to prevent the model from aggressively rushing toward a biased solution, while the shallow max_depth=3 ensures that individual models remain weak learners focused on simple patterns. By sequentially compounding these targeted adjustments, the model slowly constructs a highly robust final boundary, reducing both bias and variance to deliver strong predictive consistency on unseen testing data.

# Task4b: Feature ablation study
The feature ablation study reveals that removing the 5 lowest-importance features results in an almost identical or slightly higher test-set ROC-AUC compared to the full model, proving these dropped variables were genuinely uninformative and primarily contributing background noise. In a production setting, deploying this lower-dimensional model drastically reduces feature pipeline maintenance and shortens prediction latency (lower inference cost). This minor or nonexistent drop in performance falls well within a tolerable threshold, making the simpler model much more cost-effective and structurally stable for long-term operations.

# Task5: Cross-Validated comparison
Stratified 5-fold cross-validation provides a significantly more stable and reliable estimate of generalization performance than a single train-test split because it eliminates selection bias. A single split can randomly isolate a particularly easy or challenging subset of rows for testing, artificially distorting the performance metrics. By partitioning the dataset into five distinct folds, iteratively training on four and validating on the remaining one, every single data point serves as both training and testing data exactly once. This holistic approach averages out local statistical anomalies, ensuring the final mean and standard deviation metrics reflect how the model genuinely handles unseen data rather than just a lucky draw.

# Task6: Hyperparameter tuning with GridSearchCV
The tuning execution evaluated 90 total model configurations, derived from testing 18 unique parameter combinations (3 n_estimators × 3 max_depth × 2 min_samples_leaf) across 5 cross-validation folds. While exhaustive Grid Search guarantees discovering the absolute mathematically optimal parameter combination within the defined boundaries, it incurs a massive, linear computational cost that can easily bottleneck large workflows. In contrast, Randomized Search cuts down training times drastically by sampling a fixed number of random configurations from the parameter distribution, introducing a trade-off where minor optimization precision is willingly traded for massive gains in compute efficiency and speed.

# Task7: Manual Learning Curve
The manual learning curve evaluation highlights the structural behavior of our tuned Random Forest pipeline across five discrete data volume horizons (20% to 100%):

* Training AUC Trend: As the training fraction scales upward, the training AUC displays a gradual, expected downward trend. This confirms the classic behavior of non-linear models which easily memorize and overfit small sample volumes (e.g., at 20%) but face a more demanding, generalized learning curve as data complexity grows.
* Test AUC Trend: Conversely, the test AUC climbs continuously as more data is introduced, indicating that the model actively uses additional data points to construct more accurate decision boundaries.
* Final Capacity Conclusion: Because the test AUC is still rising at the 100% data threshold rather than flattening out entirely, the model is currently bounded by data quantity rather than mathematical model capacity. Collecting more training samples would likely yield further generalization gains, as the underlying architecture still has open capacity to absorb new patterns without hitting a performance plateau.

# Task8: Summary Comparison Table

Model	5-Fold CV Mean AUC	5-Fold CV Std Dev	Test-Set AUC
Logistic Regression (Part 2)	0.9546	0.0059	0.9521
Controlled Decision Tree (Part 3)	0.9549	0.0095	0.9515
Random Forest (Task 4)	0.9781	0.0035	0.9733
Gradient Boosting (Task 4a)	0.9769	0.0044	0.9712

The **Random Forest** model is recommended for production deployment. It achieves the highest out-of-sample performance across the board, securing a top 5-fold CV Mean AUC of 0.9781 and a Test-Set AUC of 0.9733. Crucially, it demonstrates the highest operational stability with the lowest standard deviation (±0.0035), proving it is highly resilient against data volatility and selection bias. While simpler linear and single-tree architectures offer rapid inference, the substantial ~2.1% drop in predictive accuracy makes them sub-optimal compared to the highly stable, top-performing Random Forest ensemble.
