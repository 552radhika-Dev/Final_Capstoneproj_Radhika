# Part 2: Supervised Machine Learning — Task 1

## Label and Feature Matrix Definitions

* **Feature Matrix (X):** Consists of all independent features from `cleaned_data.csv` used to predict car values, with the continuous target column (`selling_price`) and the nominal column (`name`) excluded. It contains numerical attributes (`year`, `km_driven`, `mileage`, `engine`, `max_power`, `seats`) and categorical attributes (`fuel`, `seller_type`, `transmission`, `owner`).
* **Regression Label (y_reg):** Defined as the continuous numeric variable `selling_price` representing the raw transaction cost of the used car.
* **Classification Label (y_clf):** A binary column derived by binarizing `y_reg` at its median value . Cars with a price strictly greater than the median are classified as `1` (Premium Price Tier), while cars below or equal to the median are classified as `0` (Standard Price Tier).
