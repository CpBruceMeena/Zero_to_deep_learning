import pandas as pd
import matplotlib.pyplot as plt
#from pandas.tools.plotting import scatter_matrix

df = pd.read_csv("F://PycharmProjects//Zero_to_deep_learning//titanic_train.csv")
print(df.head(10))

#_ = scatter_matrix(df.drop('PassengerId', axis = 1), figsize = (10, 10))

pd.plotting.scatter_matrix(df, alpha = 0.5)
plt.show()