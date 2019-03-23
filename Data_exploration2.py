import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("F://PycharmProjects//Zero_to_deep_learning//weight-height.csv")
df.info()
print(df.head(10))
'''
df.plot(kind = 'scatter', y = 'Weight', x = 'Height')
males = df[df['Gender'] == 'Male']
female = df[df['Gender'] == 'Female']
fig, ax = plt.subplots()

males.plot(kind = 'scatter', x = 'Height', y = 'Weight', ax = ax, color = 'red', alpha = 0.3, title = 'Male')
female.plot(kind = 'scatter', x = 'Height', y = 'Weight', ax = ax, color = 'blue', alpha = 0.3, title = 'female')

df['Gendercolor'] = df['Gender'].map({'Male' : 'red', 'Female': 'blue'})
print(df.head(10))
'''

Males = df[df['Gender'] == 'Male']
Female = df[df['Gender'] == 'Female']
'''
Males['Height'].plot(kind = 'hist', bins = 50, range = (50, 80), alpha = 0.3, color = 'blue')
Female['Height'].plot(kind = 'hist', bins = 50, range = (50, 80), alpha = 0.3, color = 'red')

plt.title('Height distribution')
plt.legend(['Males', 'Females'])
plt.xlabel("Height (in)")
'''

Males['Weight'].plot(kind = 'box', color = 'blue')
Female['Weight'].plot(kind = 'box', color = 'red')

plt.title('Weight distribution')
plt.legend(['Males', 'Females'])
plt.show()
