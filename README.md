# DQLab - Python Project : Customer Churn Prediction using Machine Learning
# Project Brackground
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

# Dataset
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
![image](https://github.com/user-attachments/assets/292cbcc2-d2ac-460e-a568-9684e8d009cc)


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


# Project Result
Full Code : [Python - Customer Churn Prediction using Machine Learning](https://github.com/oktaviorezap/Customer-Churn-Prediction-using-Machine-Learning/blob/main/(Full%20Code)%20DQLab%20-%20Customer%20Churn%20Prediction%20Using%20Machine%20Learning.ipynb)
<br>
<br> **Number of Churn Customer Before Predicted:**
<br>![image](https://github.com/user-attachments/assets/0210caae-e058-4ba0-ab97-bec531d54909)
<br>
<br>
<br> **Number of Churn Customer After Predicted:**
<br>![image](https://github.com/user-attachments/assets/1f585f18-0820-4d78-928d-546aad1f83dd)
