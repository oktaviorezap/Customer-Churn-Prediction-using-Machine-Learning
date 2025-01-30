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
![image](https://github.com/user-attachments/assets/436848e9-9f5f-4489-a74c-bdc82651f8bf)


**Model Selection Result :**
<br>
In order to choose the best model based on precision and taking into account that the model is not overfitting or underfitting, we must consider several factors:

- Precision Class 0 (No Churn Customer): Since you want to avoid False Positives in Class 1 (Churn Customer), we need to focus on the precision for Class 0.
- Precision Class 1 (Churn Customer): Precision is important, but you have already set the focus on avoiding False Positives, so Class 1 precision is also relevant.
- Comparison between Train and Test: A small gap between training and testing results indicates a model that is neither overfitting (overfit to the training data) nor underfitting (not able to capture the data pattern).
<br>

- `Gradient Boosting`:
    - Train Precision Class 0: 0.844296
    - Train Precision Class 1: 0.702980
    - Test Precision Class 0: 0.830256
    - Test Precision Class 1: 0.642857
    - Gap: Quite small between train and test for both classes, indicating the model is relatively balanced. Precision Class 0 in both train and test remains high.

- `Random Forest`:

    - Train Precision Class 0: 0.995262
    - Train Precision Class 1: 0.996868
    - Test Precision Class 0: 0.824423
    - Test Precision Class 1: 0.585421
    - Gap: Precision in train is very high compared to test, especially for Class 0. This indicates possible overfitting. The model fit the training data very well, but not well enough in the test data.

- `Logistic Regression`:

    - Train Precision Class 0: 0.832351
    - Train Precision Class 1: 0.659751
    - Test Precision Class 0: 0.827976
    - Test Precision Class 1: 0.634568
    - Gap: The gap between train and test is very small, indicating a fairly balanced model and neither overfitting nor underfitting. Precision for Class 0 in both train and test remains high.
<br>

**Conclusion:**
- `Gradient Boosting` seems to be the best model based on its high precision for Class 0 and small gap between train and test. The model is able to maintain a balance without overfitting or underfitting.
- `Random Forest` showed overfitting, with a large gap between train and test, making it less ideal to use.
- `Logistic Regression` shows good balance, although the precision for Class 0 is slightly lower than Gradient Boosting, the small gap shows the stability of this model.

So, the `Gradient Boosting` model is the best choice for this case.

# Determine Best Model (Part 2: with Additional New Models)
To Prevent **False Positive (Churn Customer predicted as No Churn Customer)**, `Precision` is the best Metrics to consider the Best Model
![image](https://github.com/user-attachments/assets/116df3ce-f6be-4a8b-8e85-de9eab176c44)


**Model Selection Result:**
<br>
From the given data, we can see a model that has a balanced performance between training and testing with a slight difference (small gap) in precision for each class (Class 0 and Class 1). We also want to make sure the model is not overfitting (fits the training data too well) or underfitting (does not learn well).

Some models that meet these criteria are:
- `LightGBM`

    - Precision Train (Class 0): 0.881627
    - Precision Train (Class 1): 0.795261
    - Precision Test (Class 0): 0.835848
    - Precision Test (Class 1): 0.640187

    - Reason:
      - Precision in train and test is quite balanced, although there is a slight decrease in test data, but the gap is not too large (thus indicating no overfitting).
      - Precision Class 0 and Class 1 are still quite good in both data sets.

- `CatBoost`

    - Precision Train (Class 0): 0.840235
    - Precision Train (Class 1): 0.697796
    - Precision Test (Class 0): 0.831461
    - Precision Test (Class 1): 0.662437

    - Reason: Precision is relatively balanced between train and test, although there is a slight decrease in test data. The gap between train and test is still acceptable.

- `Gradient Boosting`

    - Precision Train (Class 0): 0.844296
    - Precision Train (Class 1): 0.702980
    - Precision Test (Class 0): 0.830256
    - Precision Test (Class 1): 0.642857

    - Reason: Precision is fairly balanced between train and test, with a small gap between the two datasets.
<br>

By prioritizing Precision for Class 1 (Churn Customers), you aim to ensure that when a customer is predicted to churn (Class 1), the prediction is accurate and there are few False Positives (customers who actually Churn but are predicted as No Churn, Class 0).

So, in this case, you're focusing on precision for Class 1 (Churn), where you care about making fewer mistakes in predicting customers who churn, even if it means sacrificing a bit of recall or detecting fewer customers who might churn.

Given this, let's reassess the models based on Class 1 precision:
- `LightGBM` has the highest precision for Class 1 (Churn) in the training set (0.795261), and while its test precision for Class 1 is also good (0.640187), it is still one of the better choices. The gap between training and testing is not too large, indicating it is a balanced model.

- `CatBoost` also does quite well with Class 1 precision (training precision of 0.697796 and testing precision of 0.662437). But the slight drop from training to test precision is greater than LightGBM, which might signal more variability in its performance.
<br>

Thus, despite `CatBoost` being a strong contender, `LightGBM` stands out as the most balanced option for your goal of minimizing False Positives for Churn Customers.

# Prediction Result
Full Code : [Python - Customer Churn Prediction using Machine Learning](https://github.com/oktaviorezap/Customer-Churn-Prediction-using-Machine-Learning/blob/main/(Full_Code)_DQLab_Customer_Churn_Prediction_Using_Machine_Learning.ipynb)
<br>
<br> **Number of Churn Customer Before Predicted:**
<br>![image](https://github.com/user-attachments/assets/aae9755c-0663-4575-a6f5-eb4874f62c1b)
<br>
<br>
<br> **Number of Churn Customer After Predicted (Gradient Boosting):**
<br>![image](https://github.com/user-attachments/assets/2eacd050-3265-420f-b7fc-a3d8e2a4b49d)
<br>
<br>
<br> **Number of Churn Customer After Predicted (LightGBM):**
<br>![image](https://github.com/user-attachments/assets/9e770729-7063-431f-9ca5-01f793ba361e)

# Business Impact Analysis
**Business Objective** : Reducing the Number of Churn
<br>Although the percentage of churn rate has decreased after prediction (Actual Data : **26.42%**; Gradient Boosting: **19.76%**; LightGBM: **21.27%**), we also need to look at the Business Impact of various Business Metrics after Prediction which is seen from **False Positive (Churn predicted as No Churn)** and **False Negative (No Churn predicted as Churn)*** among others that is **Potential Revenue / Loss** to measure the potential Revenue / Loss by Prediction Result.

**Data Provided (Actual Data):**
1.    Average Monthly Charges (Churn Customer): $74.61 per Month
2.    Average Monthly Bill (No Churn Customers): $61.54 per Month
3.    Average Length of Stay (Churn Customers): 17.99 Months
4.    Average Tenure (Without Churn Customers): 37.61 Months
5.    Total Revenue per Month : $314,709.04 per Month

## Business Impact Analysis Implementation
<br>

1. **True Positive (TP)**: No Churn Customer (Class 0) actually Predicted as No Churn Customer (Class 0) 
2. **True Negative (TN)** : Churn Customer (Class 1) actually Predicted as Churn Customer (Class 1)
3. **False Positive (FP)**: Churn Customer (Class 1) Predicted as No Churn Customer (Class 0)
4. **False Negative (FN)**: No Churn Customer (Class 0) Predicted as Churn Customer (Class 1)
<br>

**Gradient Boosting** (True Positive : 4,662 ; True Negative: 921 ; False Positive: 915 ; False Negative: 452):
1. **Potential Revenue per Month (TP)**: **$279,258.61 per Month**
2. **Potential Loss per Month (FN)**: **$35,450.43 per Month**
3. **Potential Loss per Month (TN)**: **$73,260.60 per Month**
4. **Potential Loss per Month (FP)**: **$63,726.15 per Month**

**Light GBM**(True Positive: 4,716 ; True Negative: 1,080 ; False Positive: 756 ; False Negative: 398):
1. **Potential Revenue per Month (TP)**: **$283,650.99 per Month**
2. **Potential Loss per Month (FN)**: **$31,058.05 per Month**
3. **Potential Loss per Month (TN)**: **$84,855.05 per Month**
4. **Potential Loss per Month (FP)**: **$52,131.70 per Month**
<br>

`LightGBM` truly crowns itself as the best Model Algorithm compared to Gradient Boosting in terms of Model Performance. The Monthly Revenue Potential obtained by the Company if it applies `LightGBM` as an Algorithm Model is $283,650.99, about $4,392.38 more than `Gradient Boosting` which only obtains Potential Monthly Revenue of about $279,258.61.
<br>

### Suggestion for the Future Prediction Analysis
1. **Additional Data**: to Utilize more Business Metrics and Aspects to assess the Business Impact of each selected Best Model, additional data such as Customer Acquisition Cost (CAC), Customer Retention Cost (CRC) and other Financial Data such as Taxes, Interest, Costs (CAC and CRC are two aspects of these Costs) etc. are needed to analyse to see the Business Impact from the aspects of Net Profits, Pricing, and Strategic decisions to improve the Telco's business performance.
2. **Overcoming Class Imbalance**: Implement oversampling (such as SMOTE) or undersampling techniques to correct class imbalance.
3. **Model Optimisation**:
      - Try further hyperparameter tuning to improve model performance. Techniques such as grid search or random search can be used to find the best combination.
      - Evaluate the model using cross-validation to ensure the model does not overfit the training data.
4. **Using Feature Engineering**: Improve the features used for model training. For example, consider adding features related to customer behaviour that can provide deeper insights into the likelihood of churn.

<br>
With these steps, future churn predictions can be more accurate and more focused on customers at risk of churn.

# Business Recommendation
The following are business recommendations based on the prediction results of LightGBM and Gradient Boosting related to potential revenue and losses from False Positives (FP) and False Negatives (FN):
1. Optimizing Customer Retention (Suppressing FN)
   - Issue:
       - FN (False Negatives) are customers who do not actually churn, but are predicted to churn.
       - If left unchecked, companies can provide unnecessary incentives to customers who are not at risk of churn.

    - Recommendation:
        - **Customer Segmentation**: Identify customers who are FNs and look at their patterns. For example, do they have a history of regular payments or high loyalty?
        - **Cost-Effective Retention Strategies**: Don't be too aggressive in providing discounts or incentives to this group, but keep engaging with loyalty programs, new features, or personalized offers.
        - **Model Adjustment**: Fine-tune the model to more accurately differentiate between customers who actually churn and those who only look like they will churn.
2. Upselling and Cross-Selling Strategy (Converting FP)
   - Issue:
       - FPs (False Positives) are customers who will actually churn, but are predicted not to churn.
       - If left unchecked, they can be lost without the company realizing it because they are considered loyal customers.

    - Recommendation:
        - **Proactive Engagement**: Build a proactive communication program for FP customers, for example with emails, app notifications, or call centers to find out their needs.
        - **Upsell & Cross-Sell**: Provide additional product/service recommendations that are relevant to their habits. For example, if they subscribe to a basic plan, offer them an upgrade with a special discount.
        - **Exclusive Offers & Retention Deals**: Offer exclusive deals such as cashback, vouchers, or free trials of premium features to increase their engagement before they churn.

## Business Recommendation (Implementation)
1. Retention Program Based on Tenure
   - Customers with low tenure (1-12 months) tend to have higher churn.
   - Provide onboarding incentives such as discounts within the first month, free additional features, or education on the benefits of the service to keep customers sticking around longer.
   - For customers with high tenure (>60 months), offer loyalty rewards programs such as cashback or service upgrades at special prices.
   
   📌 Strategy:
       - Segment customers by tenure and create a special retention program.
       - Send emails or notifications with personalized offers to new and loyal customers.

2. Bundling & Cross-Selling Additional Services (StreamingTV, Internet, Phone)
   - Some customers only use PhoneService or StreamingTV without Internet.
   - Customers who have more than one service tend to be more loyal than those who only use one service.
   - Offer bundling packages (Internet + TV + Phone) at a lower price than if purchased separately.

    📌 Strategy:
       - Create discounted bundling packages for customers with limited services.
       - A/B testing to see which package customers are most interested in before broad implementation.
   
3. Price Adjustment Strategy Based on MonthlyCharges & TotalCharges
   - Customers with low monthly fees (MonthlyCharges < $30) are at risk of churn because they may find the service less useful.
   - Customers with high fees (MonthlyCharges > $100) can also churn because they feel the service is too expensive.

    📌 Strategy:
       - For low-cost customers, offer a discounted upgrade to a more complete plan to keep them loyal.
       - For high-cost customers, offer the option to downgrade certain services so they don't immediately unsubscribe.

4. PaperlessBilling & Payment Automation Analysis
   - Customers with PaperlessBilling = Yes may be more comfortable with payment automation.
   - Customers with PaperlessBilling = No could be a target to convert to auto-payment, as customers who use auto-payment tend to have lower churn.

    📌 Strategy:
       - Offer incentives such as first payment discounts for customers who enable auto-payment.
       - Send reminder notifications to customers who have not activated auto-payment. 

5. Proactive Customer Support for SeniorCitizens
   - If a pattern is found that SeniorCitizens have high churn, then a special approach can be taken to increase their engagement.
   - SeniorCitizens may need more assistance in using digital services, so educational programs or more senior-friendly customer support can help.

    📌 Strategy:
       - Prioritize customer support services for SeniorCitizen customers.
       - Provide guidance in the form of short, easily accessible videos/tutorials to help them understand the service.
