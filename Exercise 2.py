import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("F://PycharmProjects//Zero_to_deep_learning//HR_comma_sep.csv")
df.info()
print(df.head(10))
print(df.describe())

# df.hist()
plt.show()
print(df.left.value_counts()/len(df))

df['average_montly_hours_100'] = df['average_montly_hours']/100.0

#It converts categorical features into binary dummy columns
df_dummies = pd.get_dummies(df[['sales', 'salary']])

print(df.head(10))

X = pd.concat([df[['satisfaction_level', 'last_evaluation', 'number_project', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'average_montly_hours_100']], df_dummies], axis = 1).values
y = df['left'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

from keras.models import Sequential
from keras.optimizers import  Adam
from keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim= 20, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer= Adam(lr = 0.2), metrics=['accuracy'])

model.fit(X_train, y_train, epochs= 30)
y_test_pred = model.predict_classes(X_test)

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_test_pred)
labels=['False', 'True']
pred_labels = ['Predicted' + l for l in labels]
df = pd.DataFrame(cm, index = labels, columns = pred_labels)

print(classification_report(y_test, y_test_pred))

