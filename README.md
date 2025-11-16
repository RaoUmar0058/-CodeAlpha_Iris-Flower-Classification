
# 🌸 Iris Flower Classification

**Author:** Khadija Rao  
## 🚀 Project Overview

This project demonstrates a **complete machine learning workflow** for classifying the **Iris flower dataset** using multiple models:  

- **Logistic Regression**  
- **Decision Tree Classifier**  
- **K-Nearest Neighbors (KNN)**  

The project includes:  

- Data Loading & Exploratory Data Analysis (EDA)  
- Feature Scaling  
- Model Training, Hyperparameter Tuning & Cross-Validation  
- Feature Importance Analysis  
- Decision Boundary Visualization  
- Saving & Loading the Best Model  
- Making Predictions on New Samples  

---

## 🛠️ Libraries Used

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
````

---

## 📂 Project Structure

```
Iris_Flower_Classification/
│
├─ iris_classification.py      # Main Python script
├─ iris_best_model.pkl         # Saved best ML model
├─ iris_scaler.pkl             # Saved scaler
├─ requirements.txt            # Required libraries
└─ README.md                   # Project documentation
```

---

## 📊 Exploratory Data Analysis (EDA)

* **Pairplot**: Visualizes relationships between features for each species.
* **Heatmap**: Shows correlation between features.
* **Class Distribution**: Bar plot for species distribution.

---

## ⚙️ Model Training & Evaluation

* **Decision Tree**: Hyperparameter tuning with GridSearchCV for `max_depth` and `criterion`.
* **KNN**: Hyperparameter tuning for `n_neighbors` and `weights`.
* **Cross-Validation**: 5-fold CV for all models.
* **Best Model Selection**: Chosen based on highest test set accuracy.

---

## 📈 Results & Model Performance

| Model                | Test Set Accuracy |
| -------------------- | ----------------- |
| Logistic Regression  | ~0.97             |
| Decision Tree        | ~0.96             |
| **KNN (Best Model)** | **~1.00**         |

---

## 🌟 Feature Importance

For Decision Tree, feature importance is visualized:

* Petal length and petal width are the most important features for classification.

---

## 🖼️ Decision Boundary Visualization

The decision boundary is plotted using the **first two features** to show how the model separates classes.

---

## 💾 Saving & Loading the Model

* **Best Model:** `iris_best_model.pkl`
* **Scaler:** `iris_scaler.pkl`

```python
import joblib

# Load model and scaler
model = joblib.load('iris_best_model.pkl')
scaler = joblib.load('iris_scaler.pkl')

# Predict on new data
sample_data = [[5.1,3.5,1.4,0.2]]
sample_scaled = scaler.transform(sample_data)
prediction = model.predict(sample_scaled)
```

---

## 🧪 Sample Predictions

| Sample Features      | Predicted Species |
| -------------------- | ----------------- |
| [5.1, 3.5, 1.4, 0.2] | Iris-setosa       |
| [6.7, 3.1, 4.7, 1.5] | Iris-versicolor   |
| [7.2, 3.6, 6.1, 2.5] | Iris-virginica    |

---

## 📥 Installation & Usage

1. Clone the repository:

```bash
git clone <your-repo-link>
cd Iris_Flower_Classification
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the script:

```bash
python iris_classification.py
```

4. The script will:

* Train all models
* Show EDA plots
* Display accuracy, confusion matrix, and classification report
* Save the best model and scaler
* Make predictions on new samples

## Author
**Khadija Rao**
📧 Email: [raoumar0058@gmail.com]
Linkedin profile Rao Umar (www.linkedin.com/in/rao-umar-904807355)