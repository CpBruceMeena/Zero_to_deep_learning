import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv("F://PycharmProjects//Zero_to_deep_learning//international_airline_passenger.csv")
df.info()
df["Month"] = pd.to_datetime(df["Month"])
df = df.set_index("Month")
print(df.head())
df.plot()
plt.show()