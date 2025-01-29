# DQLab - Python Project : Customer Churn Prediction using Machine Learning
1. **Project Background**
2. **Dataset Description**
3. **Determine Best Model (Part 1)**
4. **Determine Best Model (Part 2: with Additional New Models)**
5. **Prediction Result**
6. **Business Impact Analysis Implementation**
7. **Prediction Result Conclusion**

# Project Background
**Business Problem:**
<br>DQLab Telco is a telecommunications company that already has many branches spread everywhere. Since its establishment in 2019, DQLab Telco has consistently paid attention to its customer experience so that customers will not be left behind.
<br>
<br> **Solution Problem:**
<br>Although only a little over 1 year old, DQLab Telco already has many customers who have switched subscriptions to competitors. The management wants to reduce the number of customers who switch (churn) by using machine learning.
<br>
<br>As a data scientist, I am asked to create the right model. In this assignment, I will do Machine Learning Modeling using June 2020 data.
<br>
<br>The steps that will be taken are:
1.    Perform Exploratory Data Analysis (EDA)
2.    Perform Data Pre-Processing
3.    Performing Machine Learning Modeling
4.    Determining the Best Model
5.    Predict the Customer Churn

# Dataset Description
**Dataset Link:** https://storage.googleapis.com/dqlab-dataset/dqlab_telco_final.csv
<br>
<br>Dataset details as follows:

1. `UpdatedAt` : Period of Data taken
2. `customerID` : Customer ID
3. `gender` : Whether the customer is a male or a female (Male, Female)
4. `SeniorCitizen` : Whether the customer is a senior citizen or not (Yes, No)
5. `Partner` : Whether the customer has a partner or not (Yes, No)
6. `tenure` : Number of months the customer has stayed with the company
7. `PhoneService` : Whether the customer has a phone service or not (Yes, No)
8. `InternetService` : Customer’s internet service provider (Yes, No)
9. `StreamingTV` : Whether the customer has streaming TV or not (Yes, No)
10. `PaperlessBilling` : Whether the customer has paperless billing or not (Yes, No)
11. `MonthlyCharges` : The amount charged to the customer monthly
12. `TotalCharges` : The total amount charged to the customer
13. `Churn` : Whether the customer churned or not (Yes, No)

# Determine Best Model (Part 1)
To Prevent **False Positive (Churn Customer predicted as No Churn Customer)**, `Precision` is the best Metrics to consider the Best Model
![image](https://github.com/user-attachments/assets/7314e0a9-558d-4197-9e87-6787ea93d0b5)


**Model Selection Result :**
<br>`Logistic Regression()` chosen as the model because to prevent False Positive (Churn Customer predicted as No Churn Customer) `Logistic Regression()` has the best **Precision** among other Models in the Testing Performance
<br>
<br>

- **Best model**: `Gradient Boosting`. Although there is a slight decrease in Class 1 precision in the test set, this model shows a more balanced performance between train and test compared to the overfitting Random Forest.

- **Avoid**: `Random Forest`, as it shows severe overfitting.
To avoid False Positive, it is necessary to ensure that the precision on Class 1 remains high, even though in some cases (like Random Forest) the test precision for Class 1 is much lower.

- So, `Gradient Boosting` is a more balanced and less overfit option.

# Determine Best Model (Part 2: with Additional New Models)
To Prevent **False Positive (Churn Customer predicted as No Churn Customer)**, `Precision` is the best Metrics to consider the Best Model
![image](https://github.com/user-attachments/assets/2a753d9c-4e7e-4dba-a150-87c72952f2c6)


**Model Selection Result:**
From the above analysis, `LightGBM` is a good choice as it has a higher precision for Class 1 (churn), a good balance between train and test, and does not show obvious signs of overfitting. The precision on the test is also quite good compared to other models.

# Prediction Result
Full Code : [Python - Customer Churn Prediction using Machine Learning](https://github.com/oktaviorezap/Customer-Churn-Prediction-using-Machine-Learning/blob/main/(Full_Code)_DQLab_Customer_Churn_Prediction_Using_Machine_Learning.ipynb)
<br>
<br> **Number of Churn Customer Before Predicted:**
<br>![image](https://github.com/user-attachments/assets/e535b023-ba95-41f7-ab98-3d577228c8be)
<br>
<br>
<br> **Number of Churn Customer After Predicted (Gradient Boosting):**
<br>![image](https://github.com/user-attachments/assets/ad609d84-1443-4bfd-ac1a-b59876acc76e)
<br>
<br>
<br> **Number of Churn Customer After Predicted (LightGBM):**
<br>![image](https://github.com/user-attachments/assets/3e83256a-2fe1-45f8-a185-c4da2414b8a7)

# Business Impact Analysis
**Business Objective** : Reducing the Number of Churn
<br>Although the percentage of churn rate has decreased after prediction (Actual Data : **26.42%**; Gradient Boosting: **19.78%**; LightGBM: **21.27%**), we also need to look at the Business Impact of various Business Metrics after Prediction which is seen from **False Positive (Churn predicted as No Churn)** and **False Negative (No Churn predicted as Churn)*** among others that is **Revenue Loss** to measure the potential loss of Average Revenue from the Prediction results.

**Data Provided:**
1.    Average Monthly Charges (Churn Customer): $74.61 per Month
2.    Average Monthly Bill (No Churn Customers): $61.54 per Month
3.    Average Length of Stay (Churn Customers): 17.99 Months
4.    Average Tenure (Without Churn Customers): 37.61 Months
5.    Total Revenue per Month : $314,709.04 per Month

## Business Impact Analysis Implementation
<br>

**Gradient Boosting**:
1. False Positive (FP): **FP (Gradient Boosting): 1836 - 1375 = 461**

2. False Negative (FN): **FN (Gradient Boosting): 5114 - 5575 = 461**

**Light GBM**:
1. False Positive (FP): **FP (LightGBM): 1836 - 1478 = 358**

2. False Negative (FN): **FN (LightGBM): 5114 - 5472 = 358**
<br>

### Revenue Loss Potential
<br>

**False Positive (FP)** means we misidentify a Churn Customer as a Non Churn customer, leading to potential lost revenue. We use Average Monthly Charges (Churn Customer) ($74.61) to calculate Revenue Loss.
- **Gradient Boosting (FP)**: Revenue Loss FP Logistic = 461 × 74.61 = **$34,395.21 per Month**
- **LightGBM (FP)**: Revenue Loss FP LightGBM = 358 × 74.61 = **$26,710.38 per Month**
<br>

**False Negative (FN)** means that we are misidentifying customers who are No Churn Customer as Churn Customer, leading to potential lost revenue. We use Average Monthly Charges (No Churn Customer) ($61.54) to calculate Revenue Loss.
- **Gradient Boosting (FN)**: Revenue Loss FN Logistic = 461 × 61.54 = **$28,369.94 per Month**
- **LightGBM (FN)**: Revenue Loss FN LightGBM = 358 × 61.54 = **$22,031.32 per Month**
<br>

**Conclusion**:
<br>

- **Revenue Loss Potential for Gradient Boosting**: **$34,395.21** + **$28,369.94** = **$62,765.15 per Month**
- **Revenue Loss Potential for LightGBM**: **$26,710.38** + **$22,031.32** = **$48,741.70 per Month**

# Prediction Result Conclusion
- **Revenue per Month** : **$314,709.04 per Month**
- **Revenue Loss Potential per Month for Gradient Boosting** = **$62,765.15 per Month**
- **Revenue Loss Potential per Month for LightGBM** = **$48,741.70 per Month**
- **Potential Revenue Earned per Month for Gradient Boosting** = **$314,709.04** - **$62,765.15** = **$251,943.89 per Month** 
- **Potential Revenue Earned per Month for LightGBM** = **$314,709.04** - **$48,741.70** = **$265,967.34 per Month**

From the Potential Losses and Revenues obtained by the Company, `LightGBM` is truly the best Model because the Potential Losses obtained are smaller and the Potential Revenues obtained are greater than the Prediction results with the `LightGBM` Algorithm Model when compared to `Gradient Boosting` Algorithm Model.

### Suggestion for the Future Prediction Analysis
1. **Additional Data**: to Utilize more Business Metrics and Aspects to assess the Business Impact of each selected Best Model, additional data such as Customer Acquisition Cost (CAC), Customer Retention Cost (CRC) and other Financial Data such as Taxes, Interest, Costs (CAC and CRC are two aspects of these Costs) etc. are needed to analyse to see the Business Impact from the aspects of Net Profits, Pricing, and Strategic decisions to improve the Telco's business performance.
2. **Overcoming Class Imbalance**: Implement oversampling (such as SMOTE) or undersampling techniques to correct class imbalance.
3. **Model Optimisation**:
      - Try further hyperparameter tuning to improve model performance. Techniques such as grid search or random search can be used to find the best combination.
      - Evaluate the model using cross-validation to ensure the model does not overfit the training data.
4. **Using Feature Engineering**: Improve the features used for model training. For example, consider adding features related to customer behaviour that can provide deeper insights into the likelihood of churn.

<br>
With these steps, future churn predictions can be more accurate and more focused on customers at risk of churn.

