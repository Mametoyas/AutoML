# AutoML with Metaheuristic Optimization 🤖

ระบบ Automated Machine Learning ที่ใช้ Metaheuristic Algorithm ในการค้นหา Model, Hyperparameters และ Preprocessing ที่เหมาะสมโดยอัตโนมัติ

*โปรเจกต์นี้เป็นส่วนหนึ่งของวิชา CP413202*

---

## วัตถุประสงค์

พัฒนาระบบ AutoML ที่สามารถ:
- ค้นหาและเลือก Model, Hyperparameters และ Feature Preprocessing ที่เหมาะสมให้อัตโนมัติ
- รองรับทั้งปัญหา Regression และ Classification
- ใช้ Metaheuristic Algorithm ในการ Optimize Search Space

---

## AutoML Search Space (Draft)

### 1. Preprocessing
- **Feature Scaling**: None, Standard Scaling, Min-Max Scaling
- **Feature Selection**: None, SelectKBest, PCA

### 2. Model Selection
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost
- Neural Network

### 3. Hyperparameters
แต่ละโมเดลมีการกำหนด Hyperparameters ที่เหมาะสม เช่น:
- Learning Rate, Number of Estimators
- Max Depth, Min Samples Split
- Regularization Parameters

### 4. Objective Function
ใช้ Fitness Function ในการประเมินประสิทธิภาพ:
- **Classification**: Accuracy, F1-Score, ROC-AUC
- **Regression**: RMSE, MAE, R²

### 5. Constraints
- Training Time Limit
- Model Complexity Constraints
- Memory Usage Constraints

---

## สมาชิก

- นางสาวกมลลักษณ์ พลกูล `663380030-4`
- นายภัทรวุธ บำรุงตา `663380288-5`
- นายจักรพรรดิ์ มั่งกูล `663380518-4`

---

<!-- ## 📦 โครงสร้างโปรเจกต์

```
AutoML/
├── src/              # Source code
├── data/             # Datasets
├── models/           # Trained models
├── notebooks/        # Jupyter notebooks
└── results/          # Experimental results
```

--- -->

<!-- ##การใช้งาน

```python
from automl import AutoMLOptimizer

# Initialize AutoML
automl = AutoMLOptimizer(task='classification')

# Fit and optimize
automl.fit(X_train, y_train)

# Predict
predictions = automl.predict(X_test)
``` -->

<!-- ---

## ผลการทดลอง

ระบบสามารถค้นหา Configuration ที่เหมาะสมได้โดยอัตโนมัติ โดยมีประสิทธิภาพใกล้เคียงหรือดีกว่าการปรับแต่งด้วยมือ

--- -->

<!-- ##เอกสารและรายงาน

รายงานฉบับเต็มประกอบด้วย:
1. Problem Definition
2. Metaheuristic Design
3. AutoML Architecture
4. Experimental Results
5. Discussion and Limitations

--- -->

