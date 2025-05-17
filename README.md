# Telco Customer Churn Prediction

## Project Overview
This project aims to predict customer churn for a telecommunications company using machine learning techniques. The project implements and compares two different approaches: a traditional machine learning model (Logistic Regression/Random Forest) and a Transformer-based model for tabular data.

## Problem Statement
Customer churn is a critical business metric for telecom companies. Predicting which customers are likely to churn helps companies take proactive measures to retain them. This project uses customer demographics, account information, and service usage data to predict customer churn.

### 🎯 Objective
The main goal is to predict whether a customer will **churn** (i.e., leave the service) based on their demographics, account information, and service usage.

## Project Structure
```
Telco Churn Prediction/
├── data/                      # Data directory
│   ├── raw/                  # contains Original dataset
│   └── processed/            # contains Processed datasets
├── notebooks/                # Jupyter notebooks from data exploration to model building and 
├── app/                     # Streamlit application
│   └── main.py             # Main Streamlit app
├── models/                  # Saved model files
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Implementation Details

### 1. Data Understanding and Preprocessing
- Exploratory Data Analysis (EDA)
- Handling missing values
- Feature engineering
- Data normalization and encoding
- Train-test split

### 2. Model Development
#### Model 1: Traditional Machine Learning
- Implementation of Logistic Regression, Random Forest, SVM, XGboost
- Hyperparameter tuning
- Cross-validation
- Feature importance analysis

#### Model 2: Transformer-based Model
- Implementation of a Transformer architecture for tabular data
- Custom attention mechanisms
- Model architecture based on "Attention is All You Need" paper
- Adaptations for tabular data

### 3. Model Evaluation
- Performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
- Model comparison
- Feature importance analysis
- Error analysis

## Futher Details

### Data cleaning
* Converted 'TotalCharges' column which is of object type to float type using pd.to_numeric() with errors parameter set to 'coerce' to parse invalid data to NaN.
* 11 missing values were found in the 'TotalCharges' column and were imputed by the mean() value.
* Data has no duplicates.

### Exploratory data analysis
1. Count plot shows the distribution of the churn rate in the data which showed an imbalance in the data.
![Imbalance Data](https://github.com/PiyushLuitel-07/Telco-Customer-Churn-Prediction-using-Traditional-ML-Algorithm-vs-Transformer-Based-Models/blob/main/images/churn%20imbalance.png)
2. Categorical features count plot insights:
    * Data is evenly distributed between the two genders; males and females, which might be useful in further analysis.
    * No information added by 'No Internet Service' or 'No Phone Service' and 'No' categories.
    --> **Replacing 'No Internet Service' and 'No Phone Service' entries with 'No'**.
    ![](https://github.com/PiyushLuitel-07/Telco-Customer-Churn-Prediction-using-Traditional-ML-Algorithm-vs-Transformer-Based-Models/blob/main/images/2_categorialfeature_countplot.png)
    ![](https://github.com/PiyushLuitel-07/Telco-Customer-Churn-Prediction-using-Traditional-ML-Algorithm-vs-Transformer-Based-Models/blob/main/images/2.png)
3. Histogram and box plot of continous features implies that:
    * No outliers exists.
    * 'TotalCharges' feature is right skewed.
    ![](https://github.com/PiyushLuitel-07/Telco-Customer-Churn-Prediction-using-Traditional-ML-Algorithm-vs-Transformer-Based-Models/blob/main/images/4_histogram_boxplot.png)
4. Scatter plot of 'MonthlyCharges' vs. 'TotalCharges' shows a positive correlation between both and also it affects the Churn rate positively.
    ![](https://github.com/PiyushLuitel-07/Telco-Customer-Churn-Prediction-using-Traditional-ML-Algorithm-vs-Transformer-Based-Models/blob/main/images/4_scatterplot.png)

### Feature encoding 
Several encoding techniques were tested on each categorical feature separately and One-Hot encoding all the categorical features gave the best results.

### Feature scaling
log transformation is very powerful in feature scaling specially with skewed data, hence, np.log1p() is applied on 'MonthlyCharges' and 'TotalCharges' features and with trials it proved giving the best results over MinMaxScaler() and StandaredScaler().

### Feature engineering
Binning 'tenure' feature into 6 ranges:
* 0-12 months --> '0-1 years'
* 12-24 months --> '1-2 years'
* 24-36 months --> '2-3 years'
* 36-48 months --> '3-4 years'
* 48-60 months --> '4-5 years'
* More than 60 months --> 'more than 5 years'

### Data imbalance
Data imbalance affects machine learning models by tending only to predict the majority class and ignoring the minority class, hence, having major misclassification of the minority class in comparison with the majority class. Hence, we use techniques to balance class distribution in the data.

Even that our data here doesn't have severe class imbalance, but handling it shows results improvement.
Using SMOTE (Synthetic Minority Oversampling Technique) library in python that randomly increasing the minority class which is 'yes' in our case.

SMOTE synthetically creates new records of the minority class by randomly selecting one or more of the k-nearest neighbors for each example in the minority class. Here, k= 5 neighbors is used. 

### Data Split
20% of the data were splitted for final testing, stratified by the 'Churn' (target) column.

## Telco Customer Churn Dataset Description
---

### 📦 Dataset Overview
- **Rows**: 7043
- **Columns**: 21

---

The dataset used in this project is obtained from [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)\
The data set includes information about:
- Customers who left within the last month – the column is called Churn
- Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies.
- Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
- Demographic info about customers – gender, age range, and if they have partners and dependents

### 🔍 Column Descriptions

| Column Name         | Description |
|---------------------|-------------|
| `customerID`        | Unique ID assigned to each customer. Not useful for modeling and can be dropped. |
| `gender`            | Customer’s gender: Male or Female. |
| `SeniorCitizen`     | Indicates if the customer is a senior citizen: 1 = Yes, 0 = No. |
| `Partner`           | Indicates if the customer has a partner: Yes or No. |
| `Dependents`        | Indicates if the customer has dependents (e.g., children): Yes or No. |
| `tenure`            | Number of months the customer has been with the company. |
| `PhoneService`      | Indicates if the customer has phone service: Yes or No. |
| `MultipleLines`     | Indicates if the customer has multiple phone lines: Yes, No, or No phone service. |
| `InternetService`   | Type of internet service: DSL, Fiber optic, or No. |
| `OnlineSecurity`    | Indicates if the customer has online security add-on: Yes, No, or No internet service. |
| `OnlineBackup`      | Indicates if the customer has online backup service: Yes, No, or No internet service. |
| `DeviceProtection`  | Indicates if the customer has device protection: Yes, No, or No internet service. |
| `TechSupport`       | Indicates if the customer has tech support: Yes, No, or No internet service. |
| `StreamingTV`       | Indicates if the customer has streaming TV: Yes, No, or No internet service. |
| `StreamingMovies`   | Indicates if the customer has streaming movies: Yes, No, or No internet service. |
| `Contract`          | Type of contract: Month-to-month, One year, or Two year. |
| `PaperlessBilling`  | Indicates if the customer uses paperless billing: Yes or No. |
| `PaymentMethod`     | Method of payment: e.g., Electronic check, Mailed check, etc. |
| `MonthlyCharges`    | Amount charged to the customer monthly. |
| `TotalCharges`      | Total amount charged to the customer (may contain missing values for new customers). |
| `Churn`             | **Target variable**: Indicates if the customer churned: Yes or No. |

---

### ✅ Notes
- **Categorical Features**: `gender`, `Partner`, `Dependents`, `Contract`, etc.
- **Numerical Features**: `tenure`, `MonthlyCharges`, `TotalCharges`
- `TotalCharges` might be stored as string due to bad formatting; needs to be converted to numeric.

## Results and Findings

- **Class Imbalance**: Slight imbalance → handled using **SMOTE**
- **Gender Distribution**: Equal distribution (Male ≈ Female)
- **Outliers**: No outliers detected in continuous variables (via boxplots)
- **Correlation**: 
  - `MonthlyCharges` and `TotalCharges` → positively correlated.
  - Customers with higher `MonthlyCharges` tend to **churn more**.
- **Tenure Insights**:
  - Churn is **very high during the first year**.
  - Very **low churn** beyond 4 years → indicates customer **loyalty increases with tenure**.

---

## Model 1: Transformer-Based Model (TabTransformer)

###  Architecture Overview
- Utilizes **self-attention layers** to capture dependencies among categorical features.
- Categorical features are **embedded** into dense vectors.
- These embeddings are **processed by Transformer blocks** (multi-head attention + feed-forward layers).
- Outputs are concatenated with normalized numerical features and passed through an MLP head.
- Trained using `Binary Cross-Entropy Loss` with the `Adam` optimizer.

### Performance
| Metric         | Class 0 (No) | Class 1 (Yes) |
|----------------|-------------|---------------|
| Precision      | 0.83        | 0.66          |
| Recall         | 0.91        | 0.50          |
| F1-Score       | 0.87        | 0.57          |

- **Accuracy**: 0.80  
- **Macro F1**: 0.72  
- **Weighted F1**: 0.79  
- **ROC AUC Score**: 0.8156 ✅

> 🔎 **Observations**:  
> - **Strong overall accuracy and precision**, especially for class 0 (non-churners).
> - Shows **significant improvement** in recall for churners ( **0.50**).
> - ROC AUC of **0.82** indicates good separation capability.
> - Still underperforms in detecting churners compared to Random Forest.

---

## Model 2: Random Forest Classifier

### ⚙️ Config
```python
RandomForestClassifier(
    max_depth=10,
    max_features='log2',
    min_samples_leaf=2,
    n_estimators=200,
    random_state=0
)
```

### 📈 Performance
| Metric         | Class 0 (No) | Class 1 (Yes) |
|----------------|-------------|---------------|
| Precision      | 0.89        | 0.54          |
| Recall         | 0.77        | 0.72          |
| F1-Score       | 0.83        | 0.62          |

- **Accuracy**: 0.76  
- **Macro F1**: 0.72  
- **Weighted F1**: 0.77  

> **Observations**:  
> - Slightly **lower overall accuracy** than Transformer (0.76 vs 0.80)
> - **Highest recall for churners (0.72)** among all models — critical for identifying at-risk customers.
> - More balanced F1 scores for both classes.
> - Offers interpretability through feature importance.

---

### 🔍 Feature Importance from Random Forest

| Rank | Feature                            | Importance |
|------|------------------------------------|------------|
| 1    | `tenure`                           | 0.1361     |
| 2    | `PaymentMethod_Electronic check`   | 0.1081     |
| 3    | `TotalCharges`                     | 0.1077     |
| 4    | `MonthlyCharges`                   | 0.0988     |
| 5    | `InternetService_Fiber optic`      | 0.0981     |
| 6    | `tenure_range`                     | 0.0866     |
| 7    | `Contract_Two year`                | 0.0802     |

> 📌 Key Insight:
> - **Tenure** is the top predictive feature — matches EDA findings.
> - Features like **payment method** and **monthly charges** significantly impact churn probability.

---

##  Recommendation

While both models perform well, the **Random Forest** model remains the **preferred choice** for churn prediction:

### ✔ Why Random Forest?
- **Higher recall (0.72) for churners** — crucial for preventing revenue loss.
- Balanced performance with interpretability.
- Useful for **feature importance analysis** to inform business strategies.

### ⚠️ Considerations for Transformer:
- Better overall accuracy, but still **lags behind in identifying churners** effectively.

> 📝 **Final Verdict**:  
> For a churn prediction task, **recall for the positive class (churn)** is most important.  
> **Random Forest** provides a **better trade-off** between recall, interpretability, and real-world utility.

### Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/telco-churn-prediction.git
cd telco-churn-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
- Visit [Kaggle Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Download and place in `data/raw/` directory ( already downloaded )

5. Run the Streamlit app:
```bash
streamlit run app/main.py
```

## Requirements
- Python 3.8+
- Key packages:
  - pandas
  - numpy
  - scikit-learn
  - tensorflow/pytorch
  - streamlit
  - plotly
  - jupyter
  - matplotlib
  - seaborn


## Future Improvements
- Implement more advanced models
- Add more features
- Improve model interpretability
- Enhance the dashboard with more interactive features
- Add API endpoints for model serving

## Author
Piyush Luitel

