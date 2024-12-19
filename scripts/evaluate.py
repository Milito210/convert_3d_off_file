import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix

# Load model and data
model = load_model('./models/model.h5')
data = np.load('./data/modelnet10.npz')
X_test, y_test = data['X_test'], data['y_test']

# Predict
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy}')

conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')
