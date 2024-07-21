# DQLab - Python Project : Customer Churn Prediction using Machine Learning
# Project Brackground
**Business Problem:**
<br>DQLab Telco is a telecommunications company that already has many branches spread everywhere. Since its establishment in 2019, DQLab Telco has consistently paid attention to its customer experience so that customers will not be left behind.
<br>
<br>Although only a little over 1 year old, DQLab Telco already has many customers who have switched subscriptions to competitors. The management wants to reduce the number of customers who switch (churn) by using machine learning.
<br>
<br>
<br> **Solution Problem:**
<br>As a data scientist, I am asked to create the right model. In this assignment, I will do Machine Learning Modeling using June 2020 data.
<br>
<br>The steps that will be taken are:
1.    Perform Exploratory Data Analysis (EDA)
2.    Perform Data Pre-Processing
3.    Performing Machine Learning Modeling
4.    Determining the Best Model

# Dataset
**Dataset Link:** https://storage.googleapis.com/dqlab-dataset/dqlab_telco_final.csv
<br>
<br>Dataset details as follows:

1. UpdatedAt Periode of Data taken
2. customerID Customer ID
3. gender Whether the customer is a male or a female (Male, Female)
4. SeniorCitizen Whether the customer is a senior citizen or not (Yes, No)
5. Partner Whether the customer has a partner or not (Yes, No)
6. tenure Number of months the customer has stayed with the company
7. PhoneService Whether the customer has a phone service or not (Yes, No)
8. InternetService Customer’s internet service provider (Yes, No)
9. StreamingTV Whether the customer has streaming TV or not (Yes, No)
10. PaperlessBilling Whether the customer has paperless billing or not (Yes, No)
11. MonthlyCharges The amount charged to the customer monthly
12. TotalCharges The total amount charged to the customer
13. Churn Whether the customer churned or not (Yes, No)

# Determine Best Model
To determine the best model to predict customer churn from the three models above, we must consider several relevant metrics: 
1. **Accuracy** on testing data
2. **ROC** (Receiver Operating Characteristic) score. 
<br>

The following is an analysis of each metric:

1. Random Forest:
        <br>- Training Accuracy: 0.995700 (99.57%)
        <br>- Testing Accuracy: 0.773100 (77.31%)
        <br>- ROC Score: 0.677300 (67.73%)

2. Gradient Boosting Classifier:
        <br>- Training Accuracy: 0.816000 (81.60%)
        <br>- Testing Accuracy: 0.793800 (79.38%)
        <br>- ROC Score: 0.691900 (69.19%)

3. Logistic Regression:
        <br>- Training Accuracy: 0.795700 (79.57%)
        <br>- Testing Accuracy: 0.792300 (79.23%)
        <br>- ROC Score: 0.691500 (69.15%)
<br>
<br>

**Analysis Testing:**

1. Accuracy on Testing Data:
        <br>- Random Forest: 0.773100 (77.31%)
        <br>- Gradient Boosting Classifier: 0.793800 (79.38%)
        <br>- Logistic Regression: 0.792300 (79.23%)
   <br>
   <br>
   From this, **Gradient Boosting Classifier** has the highest testing accuracy.
   <br>
   <br>
2. ROC Score:
        <br>- Random Forest: 0.677300 (67.73%)
        <br>- Gradient Boosting Classifier: 0.691900 (69.19%)
        <br>- Logistic Regression: 0.691500 (69.15%)
   <br>
   <br>
   **Gradient Boosting Classifier** also has the highest ROC score.
<br>
<br>

**Overfitting Testing:**
<br>
- Random Forest has a very high training accuracy (99.57%) compared to the testing accuracy (77.31%), indicating possible overfitting.
- Gradient Boosting Classifier has a modest training accuracy (81.60%) and a fairly good testing accuracy (79.38%), indicating a more balanced model.
- Logistic Regression has fairly close training accuracy and testing accuracy, but slightly lower accuracy than Gradient Boosting Classifier.
<br>
<br>

**Conclusion:**
<br>
The **Gradient Boosting Classifier** model was the best choice of these three models as it had the highest testing accuracy, highest ROC score, and showed no significant signs of overfitting. So the **Gradient Boosting Classifier** model was chosen as the best model to predict client’s repayment abilities.
# Project Result
Full Code : [Python - Customer Churn Prediction using Machine Learning](https://github.com/oktaviorezap/Customer-Churn-Prediction-using-Machine-Learning/blob/main/(Full%20Code)%20DQLab%20-%20Customer%20Churn%20Prediction%20Using%20Machine%20Learning.ipynb)
<br>
<br> **Number of Churn Customer Before Predicted:**
<br>![image](https://github.com/user-attachments/assets/0210caae-e058-4ba0-ab97-bec531d54909)
<br>
<br>
<br> **Number of Churn Customer After Predicted:**
<br>![image](https://github.com/user-attachments/assets/1f585f18-0820-4d78-928d-546aad1f83dd)
