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


### Data Set:
The provided dataset contains health-related data for individuals, which includes various measurements and medical test results. Here's an explanation of each column:

### 1. **Yas (Age)**:  
   The age of the individual in years.

### 2. **Cinsiyet (Gender)**:  
   Gender of the individual, where `1` represents male and `2` represents female.

### 3. **Boy (Height)**:  
   The height of the individual in centimeters.

### 4. **Kilo (Weight)**:  
   The weight of the individual in kilograms.

### 5. **BMI (Body Mass Index)**:  
   A calculation based on the individual's height and weight. It is used to categorize whether the person is underweight, normal weight, overweight, or obese.

### 6. **Bel_cevresi (Waist Circumference)**:  
   The circumference of the individual’s waist in centimeters, which is used to assess abdominal fat and the risk of cardiovascular diseases.

### 7. **Kalca_cevresi (Hip Circumference)**:  
   The circumference of the individual's hips in centimeters, another measurement used to evaluate body fat distribution.

### 8. **Sistolik_kan_basinci (Systolic Blood Pressure)**:  
   The top number in a blood pressure reading, measured in millimeters of mercury (mmHg). It represents the pressure in the arteries when the heart beats.

### 9. **Diyastolik_kan_basinci (Diastolic Blood Pressure)**:  
   The bottom number in a blood pressure reading, measured in mmHg. It represents the pressure in the arteries when the heart rests between beats.

### 10. **Diyabetes_mellitus (Diabetes Mellitus)**:  
   A binary indicator (`0` or `1`) that shows whether the individual has diabetes (`1` for yes, `0` for no).

### 11. **Hipertansiyon (Hypertension)**:  
   A binary indicator (`0` or `1`) that shows whether the individual has hypertension, or high blood pressure.

### 12. **Hiperlipidemi (Hyperlipidemia)**:  
   A binary indicator (`0` or `1`) that shows whether the individual has high cholesterol or other lipid disorders.

### 13. **Metabolik_sendrom (Metabolic Syndrome)**:  
   A binary indicator (`0` or `1`) showing whether the individual has metabolic syndrome, a group of risk factors that increase the likelihood of heart disease and diabetes.

### 14. **Sigara_Status (Smoking Status)**:  
   A numerical indicator representing whether the individual is a smoker (likely `0` for non-smokers and `1` for smokers).

### 15. **AST (Aspartate Aminotransferase)**:  
   A liver enzyme that is measured to evaluate liver function. High levels can indicate liver damage.

### 16. **ALT (Alanine Aminotransferase)**:  
   Another liver enzyme used to check for liver damage.

### 17. **ALP (Alkaline Phosphatase)**:  
   An enzyme related to the liver, bones, and bile ducts. Abnormal levels may suggest liver or bone disease.

### 18. **GGT (Gamma-glutamyl Transferase)**:  
   An enzyme used to assess liver function. Elevated levels may indicate liver disease or bile duct issues.

### 19. **LDH (Lactate Dehydrogenase)**:  
   An enzyme found in many body tissues. High levels can indicate tissue damage, particularly in the liver, heart, or kidneys.

### 20. **Total_bilirubin (Total Bilirubin)**:  
   A substance produced when red blood cells break down. High levels can indicate liver disease or bile duct problems.

### 21. **Direkt_bilirubin (Direct Bilirubin)**:  
   A form of bilirubin processed by the liver. Elevated levels may suggest liver issues.

### 22. **Albumin**:  
   A protein produced by the liver. Low levels can be a sign of liver or kidney disease.

### 23. **Total_kolesterol (Total Cholesterol)**:  
   The total amount of cholesterol in the blood, including both LDL ("bad" cholesterol) and HDL ("good" cholesterol).

### 24. **Trigliserit (Triglycerides)**:  
   A type of fat found in the blood. High levels can increase the risk of heart disease.

### 25. **HDL (High-Density Lipoprotein)**:  
   Known as "good" cholesterol. High levels are beneficial and help protect against heart disease.

### 26. **LDL (Low-Density Lipoprotein)**:  
   Known as "bad" cholesterol. High levels of LDL cholesterol can lead to heart disease.

### 27. **Lokosit (Leukocytes)**:  
   White blood cells that help fight infections. Elevated levels can indicate an infection or inflammation.

### 28. **Hemoglobin**:  
   A protein in red blood cells that carries oxygen. Low hemoglobin levels can indicate anemia.

### 29. **Trombosit (Platelets)**:  
   Blood cells that help with clotting. Low or high platelet levels can indicate blood disorders.

### 30. **Mean_corpuscular_volume (MCV)**:  
   The average size of red blood cells. Abnormal MCV levels can be used to diagnose types of anemia.

### 31. **Mean_platelet_volume (MPV)**:  
   The average size of platelets in the blood. This measure can help in diagnosing blood clotting issues.

### 32. **Ferritin**:  
   A protein that stores iron in the body. Low ferritin levels can indicate iron deficiency anemia.

### 33. **Glukoz (Glucose)**:  
   The amount of sugar in the blood. Elevated glucose levels can indicate diabetes.

### 34. **HemoglobinA1c**:  
   A test that measures average blood sugar levels over the past 2-3 months. It is used to diagnose and monitor diabetes.

### 35. **Target**:  
   The outcome variable. It indicates whether the individual has a particular health condition or not. It is a binary value (`0` or `1`), where `1` likely represents the presence of a condition (e.g., diabetes or hypertension) and `0` indicates the absence of the condition.

### Summary:
This dataset contains health-related information, including personal details (e.g., age, gender), physical measures (e.g., BMI, blood pressure), lab results (e.g., liver function, cholesterol), and medical conditions (e.g., diabetes, hypertension). The **target** variable is binary and could be used to predict the presence or absence of a specific health condition based on the other factors. This data is useful for building predictive models for health risks, assessing individual health, and making healthcare decisions.
