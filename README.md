# Final_Capstoneproj_Radhika
# Used Car Price Prediction
# Project Overview: End-to-End
This project implements a four-stage pipeline, transforming raw automotive data into an intelligent, LLM-powered pricing and valuation system.

# Part 1: Data Preparation & Exploration (Used Car Dataset)
Acquired and cleaned a structured used car dataset. Conducted Exploratory Data Analysis (EDA) to investigate the relationships between vehicle features (e.g., year, mileage, brand, fuel type) and market price. Handled outliers and normalized numerical features to prepare for modeling.

# Part 2: Supervised Machine Learning
Developed regression models to predict car prices. Focused on feature engineering—such as encoding categorical variables and transforming date features—to train high-performance models. Established baseline metrics using Root Mean Squared Error (RMSE) to evaluate price prediction accuracy.

# Part 3: LLM Integration & Structured Outputs
Integrated a Large Language Model (LLM) via API to provide natural language explanations for pricing. Developed a robust function to interface with the model, forcing it to output structured JSON responses that classify car valuation quality (e.g., "Overpriced," "Fair Value," "Underpriced").

# Part 4: Intelligent Valuation System with Guardrails
Implemented production-conscious guardrails, including input validation to prevent invalid car specifications and response sanitization to ensure the system provides consistent, professional pricing advice.
