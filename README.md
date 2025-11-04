# ⚙️ Support Vector Machine (SVM) Analysis

This project explores **Support Vector Machines (SVMs)** for both **classification** and **regression** tasks using the `scikit-learn` library.  
It demonstrates how kernel methods can transform data into higher dimensions for better separability and how hyperparameters like `C` and `gamma` influence model performance.

---

## 📂 Project Overview

The notebook includes:
- Linear SVM for linearly separable data
- Polynomial SVM and kernel tricks for non-linear classification
- Radial Basis Function (RBF) kernel
- Support Vector Regression (SVR)
- Hyperparameter tuning using Grid Search

---

## 🧰 Tools & Libraries

- **Python**
- **NumPy**
- **Matplotlib / Seaborn**
- **Scikit-learn**

---

## ⚙️ Workflow

### 1. Linear SVM
- Used `LinearSVC()` for binary classification.
- Controlled margin width using **regularization parameter `C`**.
- Visualized decision boundaries and margin lines.
  
  ```python
  from sklearn.svm import LinearSVC
  svm_clf = LinearSVC(C=1, loss="hinge")
  svm_clf.fit(X_scaled, y)
  svm_clf.predict([[5.5, 1.7]])
  ```

---

### 2. Polynomial Kernel SVM
- Applied `SVC(kernel="poly")` to classify non-linear datasets.
- Increased **polynomial degree** for complex boundaries.
- Demonstrated how higher degree → slower model and possible overfitting.

  ```python
  from sklearn.svm import SVC
  polynomial_svm_clf = SVC(kernel="poly", degree=3, coef0=1, C=25)
  polynomial_svm_clf.fit(X, y)
  ```

---

### 3. RBF Kernel SVM
- Used `SVC(kernel="rbf")` for efficient non-linear classification.
- Studied the effect of **gamma** and **C**:
  - High `gamma` → overfitting (tight decision boundary)
  - Low `gamma` → underfitting (wide decision boundary)

  ```python
  rbf_kernal_svm_clf = SVC(kernel="rbf", gamma=6, C=0.001)
  rbf_kernal_svm_clf.fit(X, y)
  ```

---

### 4. Hyperparameter Tuning
- Used **Grid Search** to find the optimal combination of parameters.
- Compared results across multiple kernels and regularization strengths.

  ```python
  from sklearn.model_selection import GridSearchCV
  grid_search.fit(X, y)
  ```

---

### 5. Support Vector Regression (SVR)
- Implemented **ε-SVR** to fit regression data with tolerance margins.
- Visualized:
  - Support vectors (outside ε-tube)
  - ε-tube boundaries
  - Predicted regression curve

  ```python
  from sklearn.svm import SVR
  svm_reg = SVR(kernel="rbf", C=100, epsilon=0.1)
  svm_reg.fit(X, y)
  y_pred = svm_reg.predict(X)
  ```

---

## 📊 Results & Observations

- Linear SVM performed well for separable data.  
- Polynomial and RBF kernels effectively handled **non-linear patterns**.  
- Proper tuning of `C` and `gamma` significantly improved performance.  
- SVR modeled non-linear regression smoothly while controlling errors within an **epsilon margin**.  

---

## 🧩 Key Insights

- **C** controls the trade-off between margin width and classification error.  
- **Gamma** defines how far the influence of a single training point reaches.  
- **Polynomial kernels** can model complex data but at the cost of computational efficiency.  
- **RBF kernel** is a good default choice for non-linear problems.  
- **SVR** uses the same principles for regression, balancing bias and variance through `epsilon`.  


## 📈 Future Improvements

- Implement **scaling pipelines** with `StandardScaler` and `Pipeline`.  
- Compare **SVM vs Logistic Regression** on the same dataset.  
- Visualize **decision function contours** in 3D for intuition.  
- Apply SVMs to **real-world datasets** like image classification or sentiment analysis.  

---

## 🏁 Conclusion

This notebook provides a deep dive into the **mechanics of SVMs** — showcasing linear, polynomial, and RBF kernels for classification and regression.  
It highlights the importance of kernel selection and hyperparameter tuning in achieving optimal decision boundaries.
