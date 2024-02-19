# **1-Titanic Survival Prediction Project**

## **Introduction**

Welcome to the Titanic Survival Prediction project! This project focuses on analyzing the Titanic dataset to predict passenger survival using logistic regression. The sinking of the Titanic is one of the most infamous shipwrecks in history, and with this project, we aim to explore the factors that influenced passenger survival.

## **Objective**

The primary objective of this project is to develop a predictive model that accurately determines whether a passenger survived the Titanic disaster. By examining various features such as age, sex, passenger class, and embarkation port, we aim to gain insights into the factors contributing to survival.

## **Technologies Used**

Python: Programming language used for data analysis and model building.

Libraries: Utilized various Python libraries including NumPy, Pandas, Matplotlib, Seaborn, and scikit-learn for data manipulation, visualization, and model implementation.

## **Dataset Description**

The dataset used in this project is the Titanic dataset, which provides information about passengers aboard the Titanic. It includes attributes such as PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, and Embarked.

## **Project Overview**
Data Preprocessing: Handled missing values, dropped irrelevant columns, and encoded categorical variables.

Exploratory Data Analysis: Analyzed statistical measures and visualized data to understand the distribution of survivors.

Model Building: Implemented logistic regression for binary classification.

Model Evaluation: Evaluated the model's performance using accuracy score on training and testing datasets.





 
# **2-Movie Rating Prediction With Python**

## **Introduction**

This project focuses on analyzing the IMDb Movies India dataset to gain insights into Indian movie trends and predict movie ratings using various machine learning techniques.

## **Dataset Description**

The IMDb Movies India dataset contains information about Indian movies, including features such as name, director, actors, year of release, duration, votes, and rating.

## **Exploratory Data Analysis (EDA)**

The exploratory data analysis section includes various analyses and visualizations to understand the dataset:

1. Number of Movies Released Each Year: A bar chart displaying the number of movies released each year.

2. Number of Movies Released Per Genre: A bar chart illustrating the distribution of movies across different genres.

3. Top Directors with the Most Movies: A bar chart showcasing the top directors with the highest number of movies.

4. Top Actors with the Most Movies: A bar chart displaying the top actors with the highest number of movies.

5. Movie Duration vs. Rating Scatter Plot: A scatter plot visualizing the relationship between movie duration and rating.

## **Feature Engineering**

The feature engineering section involves preparing the dataset for modeling by:

Dropping irrelevant columns such as name, director, and actors.

Converting data types and cleaning values in certain columns like year and duration.

## **Machine Learning Modeling Techniques**

The machine learning modeling section includes implementing and evaluating various regression models to predict movie ratings:

1. Linear Regression Model: Training and evaluating a linear regression model.

2. K-Nearest Neighbors (KNN) Regression Model: Tuning the number of neighbors parameter and evaluating the KNN regression model.

3. Random Forest Regression: Implementing and evaluating a random forest regression model.

4. Gradient Boosting Regression: Implementing and evaluating a gradient boosting regression model.




#**Credit Card Fraud Detection Project**

## **Introduction**

The goal of this project is to develop a machine learning model to detect fraudulent credit card transactions. We explore various steps including data loading, exploratory data analysis, preprocessing, model training, and evaluation.

## **Dataset Description**

The dataset used in this project is the credit card transaction dataset (creditcard.csv). It contains information about credit card transactions including features such as transaction amount, time, and class (0 for normal transactions and 1 for fraudulent transactions).

## **Exploratory Data Analysis (EDA)**

The exploratory data analysis section involves:

i. Loading the dataset and examining the first and last 5 rows.

ii. Checking dataset information and identifying missing values.

iii. Visualizing the distribution of normal and fraudulent transactions using a count plot.

## **Data Preprocessing**

The preprocessing steps include:

i. Separating the data into legitimate and fraudulent transactions.

ii. Handling missing values.

iii. Performing under-sampling to create a balanced dataset.

iv. Splitting the data into features and targets.

v. Standardizing the features using StandardScaler.


## **Model Training and Evaluation**

Two classification models are trained and evaluated:

### **Logistic Regression Model:**

i. Training the model using logistic regression.

ii. Evaluating the model's accuracy on both training and testing data.

iii. Calculating precision, recall, and F1-score.

### **Random Forest Classifier:**

i. Training the model using a random forest classifier.

ii. Evaluating the model's accuracy on both training and testing data.

iii. Calculating precision, recall, and F1-score.

## **Results**

### **Logistic Regression Model:**

Accuracy on Training data: 95.3%

Accuracy on Testing data: 91.9%

Precision: 96.6%

Recall: 86.7%

F1-score: 91.4%

### **Random Forest Classifier:**

Accuracy on Training data: 100%

Accuracy on Testing data: 91.9%

Precision: 96.6%

Recall: 86.7%

F1-score: 91.4%
   
