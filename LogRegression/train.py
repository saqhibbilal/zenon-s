import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression
import matplotlib.pyplot as plt


bc = datasets.load_breast_cancer()
x,y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=1234)

clf = LogisticRegression(lr=0.01)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)
    
acc = accuracy(y_pred, y_test)
print("Accuracy:", acc)    

# take first 100 samples for clean visualization
n = 100
plt.figure(figsize=(10, 4))

plt.plot(y_test[:n], label="Actual", marker='o')
plt.plot(y_pred[:n], label="Predicted", marker='x')

plt.title("Logistic Regression: Actual vs Predicted")
plt.xlabel("Sample Index")
plt.ylabel("Class (0 or 1)")
plt.legend()
plt.grid(True)

plt.show()