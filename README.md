# Week 8 & 9 Challenge: Fraud Detection - Task 1 & 2

## Project Overview
This project is part of the 10 Academy AI Mastery program's Week 8 & 9 challenge, focused on improving fraud detection in e-commerce and banking transactions. The goal is to create accurate machine learning models that can detect fraudulent transactions based on transaction data.

This repository contains the code and results for:
- **Task 1: Data Analysis and Preprocessing**
- **Task 2: Model Building and Training**

## Datasets Used
1. **Credit Card Data** (`creditcard.csv`): Contains anonymized bank transaction features and fraud labels (`Class`).
2. **Fraud Data** (`Fraud_Data.csv`): Includes e-commerce transaction data with user details, transaction timestamps, purchase values, and fraud labels (`class`).
3. **IP Address Data** (`IpAddress_to_Country.csv`): Maps IP address ranges to country names.

## Steps Completed in Task 1
1. **Data Loading**: 
   The datasets were loaded into pandas dataframes for analysis and manipulation.
   - Credit Card Data: Contains anonymized features (V1 to V28), transaction amounts, and fraud labels.
   - Fraud Data: Contains user, device, and transaction details, as well as fraud indicators.
   - IP Address Data: Contains IP address range mappings to countries.

2. **Data Cleaning**: 
   The datasets were checked for missing values and duplicates, and any duplicates were removed. No missing values were found in any of the datasets.

3. **Exploratory Data Analysis (EDA)**:
   - **Univariate Analysis**: A histogram was used to examine the distribution of `purchase_value`.
   - **Bivariate Analysis**: A box plot was created to explore the relationship between `purchase_value` and fraud (`class`).

4. **Geolocation Merging**: 
   The `Fraud_Data` was merged with the `IpAddress_to_Country.csv` dataset to provide geographic context for the transactions, based on the IP addresses.

5. **Feature Engineering**:
   - **Transaction Frequency**: The number of transactions per user.
   - **Transaction Velocity**: The time difference between the signup and purchase times.
   - **Time-Based Features**: Hour of day and day of week were extracted from the purchase timestamp.

6. **Data Normalization and Encoding**:
   - Numeric features (`purchase_value`, `transaction_velocity`) were normalized using Min-Max scaling.
   - Categorical features (`source`, `browser`, `sex`) were one-hot encoded for use in machine learning models.

## Steps Completed in Task 2
1. **Model Selection and Training**:
   Several machine learning models were trained to detect fraudulent transactions, using both the **Credit Card Data** and **Fraud Data**.
   
   - **Random Forest**: A Random Forest model was trained on the **Fraud Data** to predict fraud based on the engineered features.
   - **Logistic Regression and Gradient Boosting**: These models were trained on the **Credit Card Data** to predict fraud based on the PCA-transformed features.

2. **Model Evaluation**:
   The models were evaluated using classification metrics like **accuracy**, **precision**, **recall**, and **F1-score** to assess their performance in identifying fraudulent transactions.

   - **Random Forest** performed well on the **Fraud Data** with high accuracy in detecting fraudulent transactions.
   - **Logistic Regression and Gradient Boosting** were evaluated on the **Credit Card Data**.

3. **Saving the Model**:
   The trained **Random Forest** model was saved to the `../models` directory for future use (e.g., deploying it in an API or batch prediction).

## Running the Project

### Prerequisites
You will need the following Python packages:

```bash
pip install pandas matplotlib seaborn scikit-learn joblib

## How to Run the Code

1. **Clone this repository** to your local machine:

    ```bash
    git clone https://github.com/SolomonZinabu/week-8-9
    ```

2. **Navigate** to the project directory:

    ```bash
    cd week-8-9
    ```

3. Ensure you have the necessary datasets (`creditcard.csv`, `Fraud_Data.csv`, and `IpAddress_to_Country.csv`) in the `/data` folder relative to the task notebook.

4. **To execute Task 1** (Data Analysis and Preprocessing):

    ```bash
    jupyter notebook task-1.ipynb
    ```

5. **To execute Task 2** (Model Building and Training), including saving the Random Forest model:

    ```bash
    jupyter notebook task-2.ipynb
    ```

## Results

### Task 1 Results

- **Data Cleaning**: All datasets were free from missing values, and duplicates were removed.
- **EDA**: Various visualizations were produced to understand the relationship between features like `purchase_value` and fraud (`class`).
- **Feature Engineering**: New features such as transaction frequency, velocity, and time-based features were added to the dataset to enhance model training.
- **Geolocation**: Transactions were enriched with country information based on IP address ranges.

### Task 2 Results

- **Random Forest**: A Random Forest model was trained on the **Fraud Data** to identify fraudulent transactions, achieving strong performance with high accuracy.
- **Logistic Regression and Gradient Boosting**: These models were trained on the **Credit Card Data** and evaluated using classification metrics.
- **Model Saving**: The Random Forest model was saved for future tasks in `../models/random_forest_fraud_model.pkl`.

## Next Steps

With the trained models saved, the next step is to use these models for real-time fraud detection in future tasks. This may involve serving predictions via an API or further tuning the models for improved accuracy.
