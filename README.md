# 📘 Ensemble Learning

## 📌 Introduction

**Ensemble Learning** is a powerful machine learning paradigm where **multiple models (often called "base learners" or "weak learners")** are combined to solve the same problem.\
The idea is that a group of models working together can achieve better performance than any single model alone.

### Common Ensemble Techniques

| Type      | Description |
|-----------|-------------|
| **Bagging** | Builds multiple independent models in parallel and aggregates them. Reduces variance. *(e.g., Random Forest)* |
| **Boosting** | Builds models sequentially where each tries to correct its predecessor. Reduces bias and variance. *(e.g., XGBoost)* |
| **Stacking** | Combines multiple models using a meta-model to make the final prediction. |

---
<br><br><br>

# 🌲 Random Forest

## 📌 Introduction

**Random Forest** is an **ensemble learning method based on bagging**. It builds a collection of decision trees (each trained on a random subset of data and features), and makes predictions by **aggregating** their outputs:
- **Classification**: by **majority vote**
- **Regression**: by **averaging predictions**

## ⚙️ How It Works

- Randomly samples training data with replacement (**bootstrap sampling**).
- Builds a **decision tree** for each sample.
- At each node, it randomly selects a subset of features for splitting.
- Aggregates predictions from all trees.

## 🚧 Limitations

| Limitation                    | Description |
|-------------------------------|-------------|
| Hard to interpret             | Due to many trees. |
| Can be slow for large forests | Especially during prediction. |
| May overfit noisy datasets    | If not properly tuned. |

---

## 🔧 Key Parameters (in `RandomForestClassifier` / `RandomForestRegressor`)

| Parameter           | Description |
|---------------------|-------------|
| `n_estimators`       | Number of trees in the forest. |
| `max_depth`          | Maximum tree depth. |
| `min_samples_split`  | Minimum samples to split a node. |
| `min_samples_leaf`   | Minimum samples required at a leaf node. |
| `max_features`       | Number of features to consider when splitting. |
| `bootstrap`          | Use bootstrap samples (default: True). |
| `random_state`       | Reproducibility seed. |

### Common Methods

| Method               | Description |
|----------------------|-------------|
| `fit(X, y)`           | Train the forest. |
| `predict(X)`          | Predict class labels or values. |
| `predict_proba(X)`    | Class probabilities (classifier). |
| `score(X, y)`         | Accuracy / R². |

---

## 📏 Evaluation Metrics

- Classification: **Accuracy**, **Precision**, **Recall**, **F1 Score**, **ROC-AUC**
- Regression: **MSE**, **MAE**, **R² Score**

---

## 💡 Example Use Case

### 💉 Diabetes Patients Prediction
-------------------

### Dataset Features

This section explains the structure of the **Pima Indians Diabetes Dataset**, highlighting the key features and target label used for classification.

#### 📌 Features:

-   **Pregnancies**: Number of times pregnant

-   **Glucose**: Plasma glucose concentration

-   **BloodPressure**: Diastolic blood pressure (mm Hg)

-   **SkinThickness**: Triceps skinfold thickness (mm)

-   **Insulin**: 2-hour serum insulin (mu U/ml)

-   **BMI**: Body mass index (weight in kg / height in m²)

-   **DiabetesPedigreeFunction**: A function that scores likelihood of diabetes based on family history

-   **Age**: Age in years

#### 📌 Target:

-   **Outcome**: Binary classification label

    -   `1` = Diabetic

    -   `0` = Not diabetic

### 📌 Random Forest - Classification [Demo Code](/notebooks/random_forest_classification.ipynb)

---
<br>

### 💉 Parkinson's Disease Prediction
-------------------
This case study demonstrates how to use **Random Forest for regression tasks** --- specifically to predict **Parkinson's Disease severity** using the **UPDRS score**.

-   **UPDRS** = *Unified Parkinson's Disease Rating Scale*

    -   Measures severity of Parkinson's disease.

    -   **motor_UPDRS**: motor-related symptoms.

    -   **total_UPDRS**: includes non-motor symptoms and daily living functions.

📌 **Goal**: Predict both `motor_UPDRS` and `total_UPDRS` using patient data.

### 📊 Dataset Overview

-   **5875 records** from **42 early-stage Parkinson's patients**.

-   Each record includes:

    -   **Demographic data** (e.g., subject ID, age, sex).

    -   **Time-series info** since recording began.

    -   **Biomedical voice measurements** (16 in total).

📌 **Prediction Targets**:

-   `motor_UPDRS`

-   `total_UPDRS`

🔸 Input Features

-   **subject**: Patient ID

-   **age**: Age

-   **sex**: Categorical (male or female)

-   **test_time**: Days since the first recording.

-   **Jitter measures**: Indicators of variation in fundamental frequency.

    -   `Jitter(%)`, `Jitter(Abs)`, `Jitter:RAP`, `Jitter:PPQ5`, `Jitter:DDP`

-   **Shimmer measures**: Indicators of amplitude variation.

    -   `Shimmer`, `Shimmer(dB)`, `Shimmer:APQ3`, `APQ5`, `APQ11`, `DDA`

-   **NHR, HNR**: Noise-to-Harmonics and Harmonics-to-Noise ratio.

-   **RPDE**: Nonlinear dynamic complexity measure.

-   **DFA**: Signal fractal scaling index.

-   **PPE**: Nonlinear fundamental frequency variation measure.

These features quantify aspects of speech that can be related to Parkinson's symptoms.   

### 🔸 Target Outputs

-   `motor_UPDRS`: Motor score assessed by clinicians.

-   `total_UPDRS`: Total score including non-motor symptoms.


### 📌 Random Forest - Regression [Demo Code](/notebooks/random_forest_regression.ipynb)


---
<br><br><br>

# ⚡ XGBoost

## 📌 Introduction

**XGBoost (Extreme Gradient Boosting)** is a fast, regularized, and highly optimized implementation of the **Gradient Boosting** algorithm. It builds decision trees **sequentially**, each correcting the errors of the previous one, using **gradient descent on the loss function**.

Supports both:
- **Classification**
- **Regression**

## ⚙️ How It Works

- Minimizes loss using **gradients** (first-order and second-order).
- Adds new trees to improve upon errors (residuals).
- Employs **shrinkage**, **subsampling**, **column sampling**, and **regularization** to avoid overfitting.

## 🚧 Limitations

| Limitation                  | Description |
|-----------------------------|-------------|
| Requires parameter tuning   | To get optimal performance. |
| Less interpretable          | Especially with hundreds of trees. |
| Slower than RF in training  | Due to sequential nature. |

---

## 🔧 Key Parameters (in `XGBClassifier` / `XGBRegressor`)

| Parameter         | Description |
|------------------|-------------|
| `n_estimators`    | Number of boosting rounds. |
| `learning_rate`   | Controls contribution of each tree. |
| `max_depth`       | Maximum depth of a tree. |
| `subsample`       | Sample ratio of training data. |
| `colsample_bytree`| Feature sampling for each tree. |
| `reg_alpha`       | L1 regularization. |
| `reg_lambda`      | L2 regularization. |
| `objective`       | Loss function (e.g., `'binary:logistic'`, `'reg:squarederror'`). |

### Common Methods

| Method             | Description |
|-------------------|-------------|
| `fit(X, y)`        | Train the model. |
| `predict(X)`       | Predict labels/values. |
| `predict_proba(X)` | Probability estimates. |
| `score(X, y)`      | Accuracy / R². |

---

## 📏 Evaluation Metrics

Same as before:
- Classification: Accuracy, Precision, Recall, F1, ROC-AUC
- Regression: MSE, RMSE, MAE, R²

XGBoost also supports:
- **Early stopping**: Stop training when no improvement over a set number of rounds.

---

## 💡 Example Use Case

### 💉 Predicting Bank Customer Term Deposit Subscription

In this case study, we'll build a **classification model** to predict whether a bank customer will subscribe to a term deposit product. This is a classic business application problem.

We will demonstrate how to use **XGBoost (classifier)** to solve this banking problem, and guide you through the key steps of:

-   Feature preprocessing

-   Model training

-   Model evaluation

-   Model interpretation

### Input Features:

-   **age** (numerical): Customer's age

-   **job** (categorical): Occupation

    -   Categories include:

        -   admin (administrative staff)

        -   blue-collar (manual laborer)

        -   entrepreneur (entrepreneur)

        -   housemaid (housekeeper)

        -   management (management)

        -   retired (retired)

        -   self-employed (self-employed)

        -   services (service staff)

        -   student (student)

        -   technician (technician)

        -   unemployed (unemployed)

        -   unknown (unknown)

-   **marital** (categorical): Marital status

    -   `single`, `married`, `divorced`

        -   **education** (categorical): Education level

    -   `primary` (elementary), `secondary` (high school), `tertiary` (tertiary or higher), `unknown`

        -   **default** (categorical): Whether the client has credit card default history

    -   `yes`, `no`, `unknown`

        -   **balance** (numerical): Account balance

    -   **housing** (categorical): Whether the client has a housing loan

    -   `yes`, `no`, `unknown`

        -   **loan** (categorical): Whether the client has a personal loan

    -   `yes`, `no`, `unknown`

        -   **contact** (categorical): Type of communication contact

        -   `telephone`, `cellular`, `unknown`

-   **month** (categorical): Month of last contact

    -   **day** (numerical): Day of last contact

    -   **duration** (numerical): Duration of the last contact (in seconds)

    -   **campaign** (numerical): Number of contacts performed during this campaign

    -   **pdays** (numerical): Days since last contact (from a previous campaign)

    -   **previous** (numerical): Number of contacts before this campaign

    -   **poutcome** (categorical): Outcome of the previous campaign

    -   `success`, `failure`, `other`, `unknown`

### 📌 XGBoost - Classification [Demo Code](/notebooks/xgboost_classification.ipynb)


---
<br>

### 💉 Predicting Vehicle Fuel Efficiency

This vehicle fuel efficiency dataset comes from the **CMU StatLib** repository. It is primarily used to analyze fuel consumption in urban driving conditions.\
The dataset contains **391 records** (originally 398, but 7 were removed due to missing values).\
Each record has **7 input features**, including:

-   number of cylinders

-   engine displacement

-   horsepower

-   vehicle weight

-   acceleration

-   model year

-   and origin of manufacture

These variables are used to predict the **fuel efficiency (mpg)** of the vehicle.

* * * * *

### Input Features:

-   **cylinders**: Number of cylinders in the engine (integer).

-   **displacement**: Engine displacement (unit: cubic inches).

-   **horsepower**: Engine output power (unit: horsepower).

-   **weight**: Vehicle weight (unit: pounds).

-   **acceleration**: Time required for the vehicle to go from 0 to 60 miles per hour (unit: seconds).

-   **model year**: Year of vehicle production (integer).

-   **origin**: Region of manufacture (categorical: 1 = USA, 2 = Europe, 3 = Japan).

-   **car name**: Name of the car (string, categorical feature).

* * * * *

### Output:

-   **mpg (Miles Per Gallon)**: Fuel efficiency, i.e., the number of miles the vehicle can travel per gallon of fuel.


### 📌 XGBoost - Regression [Demo Code](/notebooks/xgboost_regression.ipynb)

---

