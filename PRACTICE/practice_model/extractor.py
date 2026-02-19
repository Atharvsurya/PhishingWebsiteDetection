from sklearn.model_selection import train_test_split
import pandas as pd


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('data/practice_data.csv')

X = data[['len','dash']]
Y = data['label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=102356)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train,Y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(Y_test,predictions)

print(f"Model Accuracy on 20-link set: {accuracy * 100}%")