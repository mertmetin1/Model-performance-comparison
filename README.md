### Model Performance Report

This report evaluates the performance of four different machine learning models (L1, L2, KNN, Random Forest) and analyzes the performance of the final model (L2) on the test data.

---

### 1. Data Analysis

The dataset contains various features used to classify the health status of patients (target). Initially, the dataset was examined, checking for missing values. Descriptive statistics (mean, standard deviation, etc.) and class distributions were observed.

---

### 2. Logistic Regression Model with L1 Regularization

#### Parameter Selection:
- Different values of **C** (`0.001`, `0.01`, `0.1`, `1`, `10`, `100`) were tested with L1 regularization to optimize model accuracy.
  
#### Results:
- **C=0.1** and **C=1** resulted in the highest accuracy (around **0.8306**).
- The accuracy of the L1 model stabilized as the **C** value increased.

![image](https://github.com/user-attachments/assets/d4bc6ba1-9a9d-4bd4-a4b6-08e6bd85f493)


---

### 3. Logistic Regression Model with L2 Regularization

#### Parameter Selection:
- Similarly, various **C** values were tested with L2 regularization.

#### Results:
- **C=0.1** provided the highest accuracy (around **0.8389**). The L2 model performed better than the L1 model.

![image](https://github.com/user-attachments/assets/95a47eb2-9dc8-4965-b93f-a7435785cf83)


---

### 4. K-Nearest Neighbors (KNN) Model

#### Parameter Selection:
- Various **n_neighbors** values (ranging from 1 to 20) were tested.

#### Results:
- **n_neighbors=11** achieved the highest accuracy (around **0.8368**). The performance of the KNN model improved as the number of neighbors increased.

![image](https://github.com/user-attachments/assets/d58e25a0-0c82-400b-9928-057d21b6b1ef)


---

### 5. Random Forest Classifier (RFC)

#### Parameter Selection:
- The **ccp_alpha** parameter was adjusted to control model complexity, preventing overfitting.

#### Results:
- **ccp_alpha=0.0** yielded the highest accuracy (around **0.8368**).

![image](https://github.com/user-attachments/assets/2e38b9d7-d2b3-45af-b695-5e7b6106995b)


---

### 6. Final Model: L2 Logistic Regression

The results indicate that the **L2 Logistic Regression** model performed the best. This model was selected with **C=0.1** and provided the best performance.

#### Test Results:
- **Test Accuracy:** 0.8389
- **ROC AUC:** 0.8974

#### Confusion Matrix:
The confusion matrix visualized the correct and incorrect classifications. The model's predictions for the positive and negative classes were evaluated.

![image](https://github.com/user-attachments/assets/23f036c1-13e7-4833-bdda-af5a4c3a709c)


#### Precision and F1 Score:
- **Precision:** 0.8294
- **F1 Score:** 0.8627

#### Recall:
- **Recall (Positive class):** 0.8281

These metrics show the model's ability to correctly predict the positive class and reduce false negatives.

---

### Conclusion and Evaluation:

The **L2 Logistic Regression Model** achieved the highest accuracy and AUC scores on the test data. This model demonstrates excellent performance in both accuracy and more important metrics like recall. The precision and F1 scores are also highly satisfactory.

Thus, the **L2 Logistic Regression** model has been selected as the best model for this dataset.



