import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('mitbih_train.csv')

sig=df.iloc[0,:-1]

plt.plot(sig)
print(df.iloc[0,-1])
plt.show()