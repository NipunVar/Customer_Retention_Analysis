# E-commerce Repeat Buyers Churn Prediction Using Machine Learning

## Table of Contents
- [Description](#description)
- [Data](#data)
- [Feature Engineering](#feature-engineering)
- [Model Fitting](#model-fitting)
- [Cross Validation Algorithm](#cross-validation-algorithm)
- [Results](#results)
- [Deep Learning Extension](#deep-learning-extension)
- [References](#references)
- [Resources](#resources)

---

## Description

### Business Interest and Background
Merchants often run large promotions (e.g., discounts or cash coupons) on major sales events like Boxing Day, Black Friday, or Double 11 (Nov 11th) to attract new buyers. Many of these buyers are one-time deal hunters, limiting the long-term impact on sales. Predicting which customers are likely to become repeat buyers helps merchants target potential loyal customers, reduce promotion costs, and improve ROI.  

---

## Data

### Data Source
The dataset is provided by Alibaba Cloud and contains anonymized user shopping logs from the six months leading up to "Double 11" along with labels indicating repeat buyers. Due to privacy concerns, the data is sampled, which may not reflect the exact statistics of Tmall.com, but it is sufficient for modeling purposes.

### Data Processing
The training dataset contains 260,864 users' data, including profile information and user activity logs. Due to the large size, the data was processed in chunks using pandas (`chunksize=10000`) to manage memory efficiently.  

### Exploratory Data Analysis
Initial EDA revealed insights into user profiles and behavior:

- **User Profile**: Demographics and basic user information  
- **User Behavior**: Purchase and browsing activity patterns  
- **Total Actions**: Overall user activity counts  
- **Action By Month**: Monthly distribution of actions  

---

## Feature Engineering

Features were generated using aggregation and grouping techniques. The final dataset includes 81 features categorized into:

1. **Action-Based Features**  
2. **Day-Based Features**  
3. **Product Diversity**  
4. **User-Merchant Similarity**  
5. **Recent Activities**  

These features were added to both the training and testing datasets.  

---

## Model Fitting

Several traditional machine learning models were evaluated to handle the imbalanced outcome:

- Random Forest  
- Logistic Regression  
- Gradient Boosting Machine  
- Extreme Gradient Boosting (XGBoost)  

To address data imbalance, the following sampling techniques were used:

- SMOTE  
- Random Under Sampler  
- ADASYN  

---

## Cross Validation Algorithm

A stratified k-fold cross-validation algorithm was implemented to handle data imbalance. It supports multiple scoring metrics from `sklearn.metrics` and allows the integration of different sampling methods. The algorithm evaluates model performance while adjusting for the imbalanced target variable.  

---

## Results

Models were evaluated using **Accuracy** and **AUC**.  

- **Best Performing Model**: XGBoost with SMOTE  
- **Observations**: The model achieved the highest accuracy and a strong AUC score. Feature importance analysis provides insights for predicting repeat buyers.  

---

## Deep Learning Extension

To improve performance, a neural network was implemented using **TensorFlow** and **Keras**:

- Added an extra hidden layer between the input and output layers  
- Handled class imbalance using initial weights, class weights, and oversampling  
- Achieved improved model performance over traditional machine learning models  

---

## References

1. Guimei Liu, Tam T. Nguyen, Gang Zhao. *Repeat Buyer Prediction for E-Commerce*. [KDD 2016](https://www.kdd.org/kdd2016/papers/files/adf0160-liuA.pdf)  
2. Rahul Bhagat, Srevatsan Muralidharan. *Buy It Again: Modeling Repeat Purchase Recommendations*. [KDD 2018](https://assets.amazon.science/40/e5/89556a6341eaa3d7dacc074ff24d/buy-it-again-modelingrepeat-purchase-recommendations.pdf)  
3. Huibing Zhang, Junchao Dong. *Prediction of Repeat Customers on E-Commerce Platform Based on Blockchain*. [Wireless Communications and Mobile Computing, 2020](https://www.hindawi.com/journals/wcmc/2020/8841437/)  
4. D. M. Blei, A. Y. Ng, M. I. Jordan. *Latent Dirichlet Allocation*. JMLR, 3(4-5):993–1022, 2003.  
5. L. Breiman. *Random Forests*. Machine Learning, 45(1):5–32, 2001.  
6. T. Chen, T. He. *XGBoost: Extreme Gradient Boosting*. [GitHub](https://github.com/dmlc/xgboost)  
7. M. Dash, H. Liu. *Feature Selection for Classification*. Intelligent Data Analysis, 1(1):131–156, 1997  

---

## Resources

Original dataset can be found [here](https://tianchi.aliyun.com/competition/entrance/231576/information)
