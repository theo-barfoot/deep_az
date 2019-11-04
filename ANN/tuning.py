from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import tensorflow.compat.v1 as tf  # required to work with tf2.0
tf.disable_v2_behavior()

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Tuning the ANN
# hyper parameters: epochs, batch size, optimiser, size of layers (number neurons)


def build_classifier(optimizer):
    _classifier = Sequential()
    _classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim = 11))
    _classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    _classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    _classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return _classifier


classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [25, 32],  # chooses two values (not a range)
              'epochs': [10, 50],
              'optimizer': ['adam', 'rmsprop']}  # dictionary of hyper-parameters
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
print(best_parameters)
best_accuracy = grid_search.best_score_
print(best_accuracy)