import pandas as pd
df = pd.read_csv('SMSSpamCollection.txt', sep = "\t", header=None, names=["mtypes", "ms"])
# print(df.head())
feature_cols = ['mtypes', 'ms']
X = df.loc[:, feature_cols]
print(X.shape)
y = df.mtypes
print(y.shape)

print(X)
