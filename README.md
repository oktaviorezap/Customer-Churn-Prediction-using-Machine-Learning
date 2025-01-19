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
![image](https://github.com/user-attachments/assets/43366c91-8d68-443e-aa78-27b8ddb3bce0)

**Model Selection Result :**
<br>`Logistic Regression()` chosen as the model because to prevent False Positive (Churn Customer predicted as No Churn Customer) `Logistic Regression()` has the best **Precision** among other Models in the Testing Performance
<br>
<br>
- To choose the best model for customer churn classification based on the data you provide, we can consider several factors, such as:
-	Precision: Precision measures how precise the model is in predicting the positive class (in this case, Class 1 - customer churn). The higher the precision, the fewer false positives the model predicts.
-	From the table you provided, here are some insights for each model:
  -	`Gradient Boosting`: Has fairly good precision for both classes in training and testing data, with higher precision for Class 0 in training and Class 0 in testing.
	- `Random Forest`: Has very high precision for both classes in the training data (especially for Class 0 and Class 1), but the precision for Class 0 and Class 1 decreases in the testing data. This indicates that the model is overfitting on the training data.
  -	`Logistic Regression`: Has fairly stable precision for both classes in training and testing data, with more balanced performance for Class 0 and Class 1 than other models.

# Determine Best Model (Part 2: with Additional New Models)
To Prevent **False Positive (Churn Customer predicted as No Churn Customer)**, `Precision` is the best Metrics to consider the Best Model
![image](https://github.com/user-attachments/assets/d2c30dc7-85a8-4c13-999d-eb12091969e4)

**Model Selection Result:**
- From the given table, the model that has the highest precision on the test set for both classes (Class 0 and Class 1) is `Gaussian Naive Bayes` on Class 0 with precision 0.869697 (86.97%) and Class 1 with precision 0.488889 (48.89%). However, the precision on Class 1 is quite low, which may indicate class imbalance.

- However, looking at the trade-off between the precision for both classes, the `CatBoost` model has a relatively balanced precision between the two classes in the test set: 0.836233 (83.62%) for Class 0 and 0.653110 (65.31%) for Class 1. The model offers quite good results in classifying both classes well.

If we look at the overall performance, `CatBoost` could be the best choice, mainly because of the balance in precision between the two classes and the better performance compared to other models, although it is not always the highest in each individual class.

# Prediction Result
Full Code : [Python - Customer Churn Prediction using Machine Learning](https://github.com/oktaviorezap/Customer-Churn-Prediction-using-Machine-Learning/blob/main/(Full_Code)_DQLab_Customer_Churn_Prediction_Using_Machine_Learning.ipynb)
<br>
<br> **Number of Churn Customer Before Predicted:**
<br>![image](https://github.com/user-attachments/assets/e535b023-ba95-41f7-ab98-3d577228c8be)
<br>
<br>
<br> **Number of Churn Customer After Predicted (Logistic Regression):**
<br>![image](https://github.com/user-attachments/assets/83a3222c-1f1a-4d8e-a1e5-b9bb249d0d71)
<br>
<br>
<br> **Number of Churn Customer After Predicted (Catboost Classifier):**
<br>![image](https://github.com/user-attachments/assets/d35a27b9-3adb-4fe4-909f-9e9aa0f12042)

# Business Impact Analysis
**Business Objective** : Reducing the Number of Churn
<br>Although the percentage of churn rate has decreased after prediction (Actual Data : **26.42%**; Logistic Regression: **19.99%**; Catboost Classifier: **19.86%**), we also need to look at the Business Impact of various Business Metrics after Prediction which is seen from **False Positive (Churn predicted as No Churn)** and **False Negative (No Churn predicted as Churn)*** among others: 
1. **Revenue Loss**: measuring the potential loss of Average Revenue from the Prediction results.
2. **CLTV (Customer Life-Time Value) Loss**: measures the potential loss of Average Revenue that can be generated from a customer during their relationship with the company from the Prediction Result.
<br>

**Data Provided:**
1.    Average Monthly Charges (Churn Customer): $74.61 per Month
2.    Average Monthly Bill (No Churn Customers): $61.54 per Month
3.    Average Length of Stay (Churn Customers): 17.99 Months
4.    Average Tenure (Without Churn Customers): 37.61 Months
5.    Total Revenue per Month : $314,709.04 per Month

## Business Impact Analysis Implementation
<br>

**Logistic Regression**:
1. False Positive (FP): **FP (Logistic Regression): 1836 - 1389 = 447**

2. False Negative (FN): **FN (Logistic Regression): 5114 - 5561 = 447**

**CatBoost Classifier**:
1. False Positive (FP): **FP (CatBoost): 1836 - 1380 = 456**

2. False Negative (FN): **FN (CatBoost): 5114 - 5570 = 456**
<br>

### Revenue Loss Potential
<br>

**False Positive (FP)** means we misidentify a Churn Customer as a Non Churn customer, leading to potential lost revenue. We use Average Monthly Charges (Churn Customer) ($74.61) to calculate Revenue Loss.
- **Logistic Regression (FP)**: Revenue Loss FP Logistic = 447 × 74.61 = **$33,411.27 per Month**
- **CatBoost Classifier (FP)**: Revenue Loss FP CatBoost = 456 × 74.61 = **$34,010.16 per Month**
<br>

**False Negative (FN)** means that we are misidentifying customers who are No Churn Customer as Churn Customer, leading to potential lost revenue. We use Average Monthly Charges (No Churn Customer) ($61.54) to calculate Revenue Loss.
- **Logistic Regression (FN)**: Revenue Loss FN Logistic = 447 × 61.54 = **$27,509.58 per Month**
- **CatBoost Classifier (FN)**: Revenue Loss FN CatBoost = 456 × 61.54 = **$28,051.04 per Month**
<br>

**Conclusion**:
<br>

- **Revenue Loss Potential for Logistic Regression**: **$33,411.27** + **$27,509.58** = **$60,920.85 per Month**
- **Revenue Loss Potential for Catboost Classifier**: **$34,101.16** + **$28,051.04** = **$62,152.20 per Month**

# Prediction Result Conclusion
- Revenue per Month : **$314,709.04 per Month**
- **Revenue Loss Potential for Logistic Regression** = **$60,920.85 per Month**
- **Revenue Loss Potential for Catboost Classifier** = **$62,152.20 per Month**
- **Potential Revenue Earned for Logistic Regression** = **$314,709.04** - **$60,920.85** = **$253,788.19 per Month** 
- **Potential Revenue Earned for Catboost Classifier** = **$314,709.04** - **$62,152.20** = **$252,556.84 per Month**

Although the `Catboost Classifier` has a better precision for Testing for each class, the Business Impact of the `Logistic Regression` model is better because the potential lost revenue is not as large as the `Catboost Classifier`.

### Suggestion for the Future Prediction Analysis
1. **Additional Data**: In order to be able to assess the Business Impact of each selected Best Model, additional data such as Customer Acquisition Cost (CAC), Customer Retention Cost (CRC) and other Financial Data such as Taxes, Interest, Costs etc. are needed to analyse to see the Business Impact from the aspects of Net Profits, Pricing, to Strategic decisions to improve the Telco's business performance.
2. **Predicted Distribution vs Original Data**:
   - `Logistic Regression`: This model predicts 5561 customers as No Churn and 1389 customers as Yes Churn. From this result, there is a slight difference compared to the original data (5114 No Churn and 1836 Yes Churn).
   - `CatBoost Classifier`: This model predicts 5570 customers as No Churn and 1380 customers as Yes Churn. Here, the prediction is slightly closer to the actual number of customers who did not churn (5114), but there is still a slight difference.

3. **Expected Number of Customers to Churn (Yes)**:
   - The original data shows 1836 churn (Yes) customers. Both models predict a lower number of churns, 1389 for `Logistic Regression` and 1380 for `CatBoost Classifier`. This could indicate that both models are more likely to predict customers as No Churn compared to Yes Churn.

4. **Imbalance**:
   - In the original data, churn is the minority (only 1836 out of 6950), while no churn is the majority (5114 out of 6950). This reflects class imbalance, which is a challenge in classification, as models tend to predict the majority class (No Churn) more often to minimise overall error.
   - In this case, although both models (`Logistic Regression` and `CatBoost Classifier`) tended to predict No Churn more, `CatBoost Classifier` was slightly more balanced in the number of churn predictions (1380) than `Logistic Regression (1389)`.

5. **Conclusion**:
   - Both models are not completely accurate in predicting churn (Yes) because there is a class imbalance in the original data. They tend to predict No Churn more often.
   - `CatBoost Classifier` is slightly closer to the original number of churn customers than `Logistic Regression`, but both models still over-predict No Churn.
   - To improve performance, you can consider techniques such as `SMOTE (Synthetic Minority Over-sampling Technique)` or using a penalty for churn misprediction in the model to address this class imbalance issue.

