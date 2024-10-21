Predicting Customer Churn Using K-Nearest Neighbors (KNN)
Overview
This project was developed as part of the CS506 Data Science Tools and Applications course. The goal of this assignment is to build a predictive model to identify customers likely to churn in the banking industry using the K-Nearest Neighbors (KNN) algorithm.

Customer churn prediction is critical for banks to retain customers by proactively identifying those at risk of leaving. This project focuses on creating a custom KNN classifier from scratch to predict customer churn based on various customer attributes like demographics, account balances, and more.

Key Features
KNN Implementation from Scratch: The entire KNN algorithm, including distance metric calculations and prediction logic, was implemented manually without using any pre-built machine learning libraries like scikit-learn.
Hyperparameter Tuning: Explored and optimized key parameters such as the number of neighbors (K) and distance metrics (Euclidean and Manhattan) to improve model performance.
Cross-Validation and Model Evaluation: Used cross-validation techniques to ensure model robustness and avoid overfitting. The model was evaluated based on the Area Under the ROC Curve (AUC).
Top Performance: Achieved a top 10 score out of 180 participants on Kaggle with an AUC of 0.91, demonstrating the effectiveness of the custom KNN implementation and tuning strategy.
Table of Contents
Installation
Dataset
Implementation Details
Results
Usage
Contributing
License

Installation
To run this project, you need Python 3.x and the following packages installed:
pip install numpy pandas

Dataset
The dataset contains the following customer attributes:

Customer ID
Credit Score
Geography (France, Spain, Germany)
Gender
Age
Tenure (number of years with the bank)
Balance
Number of Products
Credit Card Holder (Yes/No)
Active Member (Yes/No)
Estimated Salary
Exited (1 = churned, 0 = retained)
The dataset is split into training and test sets. The task is to predict the probability of churn for each customer in the test set.

Implementation Details
1. Data Preprocessing:
Handled missing values and scaled all numerical features.
Performed feature engineering to enhance model accuracy, including analysis of categorical variables (e.g., Geography and Gender) and numerical variables (e.g., Credit Score, Age).
2. KNN Algorithm:
Implemented KNN from scratch, supporting both Euclidean and Manhattan distance metrics.
For each customer, the model calculates the distance to every other customer in the training set and selects the K-nearest neighbors to classify whether the customer will churn.
3. Model Tuning:
Experimented with various values of K (number of neighbors) and distance metrics.
Selected the optimal configuration using cross-validation.
4. Model Evaluation:
Evaluated the model using AUC as the primary metric, along with accuracy, precision, and recall.
Performed 5-fold cross-validation to ensure the model's robustness and generalization.
5. Hyperparameter Tuning:
Automated the search for the best K and distance metric to maximize AUC using cross-validation.
Results
Final Model: KNN with K=20 and Euclidean distance metric.
Performance: Achieved an AUC score of 0.91, placing in the top 10 out of 180 participants on Kaggle.
Output: The model generates a prediction for each customer in the test set, indicating the probability of churn.
Usage
To run the project:

Clone this repository.
Ensure the necessary dependencies are installed.
Run the notebook or Python script that contains the KNN implementation and preprocessing steps.

Contributing
Contributions are welcome! If you have suggestions or improvements, please submit a pull request or open an issue.

License
This project is licensed under the MIT License - see the LICENSE file for details.