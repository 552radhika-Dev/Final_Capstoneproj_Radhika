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
