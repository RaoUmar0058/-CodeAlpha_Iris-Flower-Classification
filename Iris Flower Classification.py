# ===========================
# Iris Flower Classification - Complete Project (Fixed Version)
# ===========================

# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib.colors import ListedColormap
import joblib
import copy

# ===========================
# 1️⃣ Load Dataset
# ===========================
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y

print("First 5 rows of dataset:")
print(df.head())

# ===========================
# 2️⃣ Exploratory Data Analysis
# ===========================
# Pairplot
sns.pairplot(df, hue='species')
plt.show()

# Heatmap of correlations
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

# Class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='species', data=df)
plt.title("Class Distribution")
plt.show()

# ===========================
# 3️⃣ Split Data & Scale Features
# ===========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===========================
# 4️⃣ Train Multiple Models
# ===========================
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier()
}

results = {}
fitted_models = {}
best_model = None

for name, model in models.items():
    if name == "Decision Tree":
        # Hyperparameter tuning for Decision Tree
        param_grid = {'max_depth':[2,3,4,5], 'criterion':['gini','entropy']}
        grid = GridSearchCV(model, param_grid, cv=3)
        grid.fit(X_train_scaled, y_train)
        model_used = grid.best_estimator_
        print(f"{name} Best Params: {grid.best_params_}")
    else:
        model.fit(X_train_scaled, y_train)
        model_used = model

    # Predict and evaluate
    y_pred = model_used.predict(X_test_scaled)
    results[name] = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {results[name]:.2f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Store fitted model
    fitted_models[name] = model_used

    # Update best model
    if best_model is None or accuracy_score(y_test, y_pred) > accuracy_score(y_test, best_model.predict(X_test_scaled)):
        best_model = model_used

# ===========================
# 5️⃣ Cross-Validation
# ===========================
print("\n--- Cross-Validation Accuracy ---")
for name, model_used in fitted_models.items():
    scores = cross_val_score(model_used, X_train_scaled, y_train, cv=5)
    print(f"{name}: Mean CV Accuracy = {np.mean(scores):.3f}")

# ===========================
# 6️⃣ KNN Hyperparameter Tuning
# ===========================
knn_param_grid = {'n_neighbors':[3,5,7,9], 'weights':['uniform','distance']}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5)
knn_grid.fit(X_train_scaled, y_train)
knn_best = knn_grid.best_estimator_
print("\nKNN Best Params:", knn_grid.best_params_)

if accuracy_score(y_test, knn_best.predict(X_test_scaled)) > accuracy_score(y_test, best_model.predict(X_test_scaled)):
    best_model = knn_best
    print("Best model updated to KNN.")

# ===========================
# 7️⃣ Compare Model Performance
# ===========================
plt.figure(figsize=(6,4))
plt.bar(results.keys(), results.values(), color=['skyblue','lightgreen','salmon'])
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.show()

# ===========================
# 8️⃣ Feature Importance (Decision Tree only)
# ===========================
if isinstance(best_model, DecisionTreeClassifier):
    feature_importances = pd.Series(best_model.feature_importances_, index=iris.feature_names)
    feature_importances.sort_values().plot(kind='barh', color='teal', figsize=(6,4))
    plt.title("Feature Importance - Decision Tree")
    plt.show()

# ===========================
# 9️⃣ Decision Boundary Visualization (first 2 features)
# ===========================
plot_model = copy.deepcopy(best_model)
X_plot = X_train_scaled[:, :2]
y_plot = y_train
plot_model.fit(X_plot, y_plot)  # only for plotting

x_min, x_max = X_plot[:,0].min()-1, X_plot[:,0].max()+1
y_min, y_max = X_plot[:,1].min()-1, X_plot[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min,x_max,0.02), np.arange(y_min,y_max,0.02))
Z = plot_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(7,5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF']))
plt.scatter(X_plot[:,0], X_plot[:,1], c=y_plot, edgecolor='k', cmap=ListedColormap(['red','green','blue']))
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Decision Boundary (First 2 Features)")
plt.show()

# ===========================
# 🔟 Confusion Matrix Heatmap for Best Model
# ===========================
y_pred_best = best_model.predict(X_test_scaled)
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_best), annot=True, fmt='d', cmap='Blues')
plt.title("Best Model Confusion Matrix")
plt.show()

# ===========================
# 1️⃣1️⃣ Save Best Model & Scaler
# ===========================
joblib.dump(best_model, 'iris_best_model.pkl')
joblib.dump(scaler, 'iris_scaler.pkl')
print("Best model and scaler saved successfully!")

# ===========================
# 1️⃣2️⃣ Predict on New Samples
# ===========================
sample_data = np.array([
    [5.1,3.5,1.4,0.2],
    [6.7,3.1,4.7,1.5],
    [7.2,3.6,6.1,2.5]
])
sample_scaled = scaler.transform(sample_data)
predictions = best_model.predict(sample_scaled)

species_map = {0:'Iris-setosa',1:'Iris-versicolor',2:'Iris-virginica'}
for i,pred in enumerate(predictions):
    print(f"Sample {i+1}: Predicted Species -> {species_map[pred]}")
