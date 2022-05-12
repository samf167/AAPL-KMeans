import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

import Data_Import
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Import Data(2y AAPL Daily)
df = pd.json_normalize(Data_Import.df)
# imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# df = imp.fit_transform(Df)
print(df.head())

# Create features
df['Ma_3'] = df['c'].shift(1).rolling(window=3).mean()
df['Change'] = df['o'] - df['c']
df['Range'] = df['h'] - df['l']

# Create model
X = df[['Ma_3', 'Change', 'Range']]
Y = np.where(df['c'].shift(-1) > df['c'], 1, 0)

# Split dset
split_percentage = 0.8
split = int(split_percentage*len(df))

# Train data set
X_train = X[:split]
Y_train = Y[:split]

# Test data set
X_test = X[split:]
Y_test = Y[split:]

# Setup and test model
model = KMeans(n_clusters=2)
p = model.fit(X_train, Y_train)

# Report Accuracy
accuracy_train = accuracy_score(Y_train, p.predict(X_train))
accuracy_test = accuracy_score(Y_test, p.predict(X_test))

print('\nTrain Accuracy:{: .2f}%'.format(accuracy_train*100))
print('Test Accuracy:{: .2f}%'.format(accuracy_test*100))






