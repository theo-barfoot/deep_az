# Artificial Neural Network
import numpy as np
import pandas as pd

# Part 1 - Data Preprocessing
# Importing the dataset
dataset = pd.read_csv('ANN/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]  # To avoid dummy variable trap
# only want two dummy variables for the country, which has 3 options

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
# -------------------------------------------------------------------------------------------#

# import tensorflow as tf
import tensorflow.compat.v1 as tf  # required to work with tf2.0
tf.disable_v2_behavior()

# Not sure you want to include this: (I added it a while ago to see how to use GPU)
# fairly certain it is not needed as CUDA is set up correctly
# import keras
# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
# input dim specifies the size of the input layer
# units specifies the size of the hidden layer
# rule of thumb says that having this as average of input and output size (11/2)
# kernel_inituializer initialilses the weights -- uniform distribution
# initialised weights are close to but not zero

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# if you had more than two output classes, ie 3, then you would need to change
# the units to 3 and the activation to softmax

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# binary cross entropy is a logarithmic loss function for output of two classes
# if you have 3 or more output classes then choose categorial_crossentropy
# logarithmic loss function applies to sigmoid output activation?

# Fitting the ANN to the Training set
print('---------------TRAINING---------------')
classifier.fit(X_train, y_train, batch_size = 10, epochs = 10)
# batch size, number of observations forward propagated through network
# before updating the weights using back propagation
# epochs, number of times the entire dataset is passed through the network


# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

# X_hw = dict.fromkeys(dataset.columns[3:13])
# X_hw['Geography'] = 0  # France
# X_hw['CreditScore'] = 600
# X_hw['Gender'] = 1 # Male
# X_hw['Age'] = 40
# X_hw['Tenure'] = 3
# X_hw['Balance'] = 60000
# X_hw['NumOfProducts'] = 2
# X_hw['HasCrCard'] = 1  # yes
# X_hw['IsActiveMember'] = 1  # yes
# X_hw['EstimatedSalary'] = 50000

X_hw = [0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]
X_hw = np.array(X_hw)
X_hw = X_hw.reshape(1, -1)
X_hw = sc.transform(X_hw)

y_hw_pred = classifier.predict(X_hw)
y_hw_pred > 0.5
