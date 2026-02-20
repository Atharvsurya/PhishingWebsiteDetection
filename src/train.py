import pandas as pd
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('data/processed_dataset_v2.csv')
X = data.drop(['label'],axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
joblib.dump(lr,'models/temp/logistic_v2.pkl')

lr_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, lr_pred)
print(f"LogisticRegression Accuracy: {accuracy * 100:.2f}%")
print("\nLogisticRegression Classification Report:\n", classification_report(y_test, lr_pred))

#Random Forest
rf = RandomForestClassifier(n_estimators= 100)
rf.fit(X_train,y_train)
joblib.dump(rf,'models/temp/random_forest_v2.pkl')

rf_pred = rf.predict(X_test)
print(f"RandomForest Accuracy: {accuracy_score(y_test, rf_pred) * 100:.2f}%")
print("\nRandomForestClassification Report:\n", classification_report(y_test, rf_pred))

# SVM
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
joblib.dump(svm,'models/temp/svm_v2.pkl')

svm_pred = svm.predict(X_test)
print(f"SVM Accuracy: {accuracy_score(y_test, svm_pred) * 100:.2f}%")
print("\nSVM Classification Report:\n", classification_report(y_test, svm_pred))

#ChatGpt
import matplotlib.pyplot as plt
import seaborn as sns

lrcm = confusion_matrix(y_test, lr_pred)
plt.figure(figsize=(6,4))
sns.heatmap(lrcm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for LogisticRegression')
plt.savefig("models/temp.png")
plt.close()

rfcm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(6,4))
sns.heatmap(rfcm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for RandomForest')
plt.savefig("models/temp.png")
plt.close()

svmcm = confusion_matrix(y_test, svm_pred)
plt.figure(figsize=(6,4))
sns.heatmap(svmcm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for SVM')
plt.savefig("models/temp.png")
plt.close()