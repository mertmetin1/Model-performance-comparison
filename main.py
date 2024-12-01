import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, f1_score
import seaborn as sns

data = pd.read_csv("HW2DataSet2024.csv")

print(data.head())
print(data.info())
print(data.describe())
print("null check :", data.isnull().sum())
X = data.drop("target", axis=1)
y = data["target"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("-------------------------------------------------------------- L1 ------------------------------------------------------------------")



C_values = [0.001,0.01, 0.1, 1, 10, 100]
l1_results = []
for C in C_values:
    l1_model = LogisticRegression(penalty='l1', solver='liblinear', C=C, random_state=42)
    l1_scores = cross_val_score(l1_model, X_train_scaled, y_train, cv=5)
    l1_results.append((C, l1_scores.mean()))

print("L1Skore:")
for C, score in l1_results:
    print(f"C={C}: ort acc={score}")

l1_scores = [score for _, score in l1_results]
plt.figure(figsize=(10, 6))
plt.plot(C_values, l1_scores, label="L1 regulated", marker='o')
plt.xscale('log')  
plt.xlabel("C")
plt.ylabel("ort acc")
plt.title("L1")
plt.grid(True)
plt.legend()
plt.show()

#l1     C=0.1: ort acc=0.8306
#       C=1: ort acc=0.8306

print("---------------------------------------------------- L2 ----------------------------------------------------------------------------")





l2_results = []

for C in C_values:
    l2_model = LogisticRegression(penalty='l2', solver='lbfgs', C=C, random_state=42, max_iter=1000)
    l2_scores = cross_val_score(l2_model, X_train_scaled, y_train, cv=5)
    l2_results.append((C, l2_scores.mean()))

print("L2 Skoru")
for C, score in l2_results:
    print(f"C={C}: ort acc={score}")


l2_scores = [score for _, score in l2_results]

plt.figure(figsize=(10, 6))
plt.plot(C_values, l2_scores, label="L2 regulated", marker='o')
plt.xscale('log')  
plt.xlabel("C")
plt.ylabel("ort acc")
plt.title("L2")
plt.grid(True)
plt.legend()
plt.show()

#l2 C=0.1: mean acc=0.8389




print("---------------------------------------------------- KNN ----------------------------------------------------------------------------")



neighbor_values = range(1, 21) 
scores = []

for n in neighbor_values:
    knn_model = KNeighborsClassifier(n_neighbors=n)
    knn_scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=5)
    scores.append(knn_scores.mean())
    print(f"(n_neighbors)={n}, mean acc: {knn_scores.mean()}")


plt.figure(figsize=(10, 6))
plt.plot(neighbor_values, scores, marker='o', linestyle='-', color='b')
plt.xlabel("(n_neighbors)")
plt.ylabel("mean acc")
plt.grid(True)
plt.xticks(neighbor_values)
plt.show()

#KNN (n_neighbors)=11,mean acc: 0.8368


print("---------------------------------------------------- RFC----------------------------------------------------------------------------")



ccp_alpha_values = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0]
scores = []

for ccp_alpha in ccp_alpha_values:
    rfc_model = RandomForestClassifier(random_state=42, n_estimators=100, ccp_alpha=ccp_alpha)
    rfc_scores = cross_val_score(rfc_model, X_train_scaled, y_train, cv=5)
    scores.append(rfc_scores.mean())
    print(f"ccp_alpha={ccp_alpha},mean acc: {rfc_scores.mean()}")

plt.figure(figsize=(10, 6))
plt.plot(ccp_alpha_values, scores, marker='o', linestyle='-', color='r')
plt.xlabel("ccp_alpha")
plt.ylabel("mean acc")
plt.title(" ccp_alpha range 0.0 to 2.0 RFC")
plt.grid(True)
plt.show()



# ccp_alpha=0.0 mean: 0.8368


print("---------------------------------------------------- Final model----------------------------------------------------------------------------")


#seleted best model as L2


final_l2_model = LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, random_state=42, max_iter=1000)
final_l2_model.fit(X_train_scaled, y_train)

y_pred = final_l2_model.predict(X_test_scaled)
y_pred_prob = final_l2_model.predict_proba(X_test_scaled)[:, 1] 


accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print("test acc: ",accuracy)
print("test roc/auc:",roc_auc)



conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("tahmin")
plt.ylabel("Gerçek")
plt.title("confusion matrix")
plt.show()

print("confusion matrix:")
print(conf_matrix)



precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"precision:",precision)
print(f"F1:",f1)


TP = conf_matrix[1, 1]  
FN = conf_matrix[1, 0]  

recall_positive = TP / (TP + FN)
print(f"Recall :",recall_positive)
