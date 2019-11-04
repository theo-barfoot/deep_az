from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow.compat.v1 as tf  # required to work with tf2.0
tf.disable_v2_behavior()

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)


def build_classifier():
    _classifier = Sequential()
    _classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    _classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    # _classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    _classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    _classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return _classifier


classifier = KerasClassifier(build_fn=build_classifier, batch_size=32, epochs=500)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=1)
# cv = number of folds in k-fold cross validation, 10 is a very common choice

mean = accuracies.mean()  # = 0.83 = 83%, therefore low bias, high accuracy
print(mean)
variance = accuracies.std()  # = 0.009 = 0.9%, therefore low variance
print(variance)
# high variance in k-fold cross validation will imply overfitting
# it may therefore be a good idea to use drop-out regularisation
