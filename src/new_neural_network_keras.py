import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD



directory= 'data\\train'


# Loop over all the pkl files in the directory, open them and append all of them into a dataframe made of arrays.
data = pd.DataFrame()
target = pd.DataFrame()

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".pkl"):
        print(os.path.join(directory, filename))
        pkl_file = open(os.path.join(directory, filename), 'rb')
        event = pickle.load(pkl_file)
        data_, target_ = event[0], event[1]
        data_ = pd.DataFrame(data_)
        target_ = pd.DataFrame(target_)
        data = data.append(data_)
        target = target.append(target_)
        continue
    else:
        continue

# Convert the data dataframe to a list and then to an array of shape (size, 10, 10, 1)
data.reset_index(inplace=True, drop=True)
data = data[0].tolist()
data = np.array(data)
data = data.reshape(-1,10,10,1)

# Convert the target to numerical from 0 to 4:
dic_types={11: 0, 13 : 1, 211:2, 321:3, 2212 : 4}
target = target.replace(dic_types)
# Convert the target into an array of dummies per category
target.reset_index(inplace=True, drop=True)
target = target[0].tolist()
target = np_utils.to_categorical(target)

# Subsetting train and validation

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.10, random_state=42)



model = Sequential()
model.add(Conv2D(10, (1, 1), activation='relu', input_shape=(10, 10, 1), data_format=("channels_last")))
model.add(BatchNormalization()),
model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(MaxPooling2D(2,2)),
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(5, activation='sigmoid'))
model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['categorical_accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=5, batch_size = 256, verbose=2, validation_data=(X_test, y_test))



# Apply the results to the test:
pkl_file = open(os.path.join('data\\test', 'data_test_file.pkl'), 'rb')
event_test = pickle.load(pkl_file)
data_test = pd.DataFrame(event_test)
data_test = pd.DataFrame(data_test[1])
data_test = data_test[1].tolist()
data_test = np.array(data_test)

data_test = data_test.reshape(-1, 10,10,1)


results_test = model.predict(data_test)

# Rename the columns and the index
col_names={0: "electron", 1 : "muon", 2:"pion", 3:"kaon", 4 : "proton"}
results_test = pd.DataFrame(results_test)
results_test.columns = results_test.columns.map(col_names)
results_test.index = results_test.index.set_names(['image'])

# WRITE CSV FILE
results_test.to_csv('submission_keras.csv')
