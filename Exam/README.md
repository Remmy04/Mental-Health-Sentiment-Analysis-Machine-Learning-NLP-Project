# 🧪 DAML Exam Project – Breast Cancer Classification (Random Forest)

This folder contains my **Individual Exam Project** for the *Data Analytics & Machine Learning (DAML)* module.  

The task focuses on building a complete **Machine Learning classification pipeline** using the **Breast Cancer Wisconsin Dataset**, implemented entirely in **Python (Google Colab)**.

The project demonstrates practical skills in **data preprocessing**, **feature selection**, **model training**, **evaluation**, and **interpretation of ML results** using a **Random Forest Classifier**.

---

## 🧠 Project Overview

### 🎯 **Objective**
Build a machine learning model to classify tumors as **Malignant (0)** or **Benign (1)** using:

- Random Forest Classifier  
- Train/test split  
- MinMaxScaler  
- Model evaluation metrics  
- Confusion matrix  
- Feature importance visualization  

---

## 🔧 Technologies Used

- **Python**
- **NumPy, Pandas** (data processing)
- **Matplotlib, Seaborn** (visualization)
- **Scikit-learn**  
  - RandomForestClassifier  
  - train_test_split  
  - LabelEncoder  
  - MinMaxScaler  
  - Accuracy & confusion matrix  

All code is written inside the `.ipynb` notebook.

---

## 🧹 Data Preparation

Steps performed 

- Loading the dataset with `pd.read_csv()`  
- Viewing first rows to understand structure  
- Encoding the `diagnosis` label (M → 0, B → 1)  
- Separating **X** (all features) and **y** (label)  
- Scaling numerical features using `MinMaxScaler()`  

---

## 🤖 Model Building — Random Forest

The Random Forest model was built using:

- `train_test_split(test_size=0.2, random_state=42)`  
- `RandomForestClassifier(n_estimators=100)`  

The model was trained on **80% training**, **20% testing**.

---

## 📊 Model Evaluation

### **✔ Accuracy**
Extracted from exam output

- **Training Accuracy:** 1.000  
- **Testing Accuracy:** 0.965  

### **✔ Interpretation**
- Perfect training accuracy suggests the model fits the training data extremely well.  
- Testing accuracy of 0.965 confirms strong generalization.  
- Slight signs of **overfitting** appear due to 1.000 training accuracy.

### **✔ Confusion Matrix**
Based on exam results 

- Only **1 false positive**  
- Only **3 false negatives**  
- Model performs slightly better on detecting *malignant* cases  

In medical applications, reducing **false negatives** is critical to avoid missed diagnoses.

---

## 📈 Feature Importance Visualization

A horizontal bar chart shows which features contribute most to classification  
— generated using `rf_model.feature_importances_` 

Key insights:

- Mean concave points  
- Worst perimeter  
- Worst concavity  
- Worst radius  

were among the most influential predictors.

---

## 🧠 What I Learned

- Building complete ML workflows from raw data → model evaluation  
- Understanding bias, overfitting, feature importance  
- Applying ML in healthcare use cases  
- Evaluating classification performance responsibly  
- Producing clear ML reports under exam conditions  

---

✨ *This exam demonstrates my ability to design, implement, and evaluate machine learning models effectively using Python and scikit-learn.*


