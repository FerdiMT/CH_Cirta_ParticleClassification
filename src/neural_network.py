import numpy as np
import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

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

# Preprocessing the data and targets
data = data[0].tolist()
# Switch to numeric 0-4 (maybe there is another way for categorical data into NN)
target = target.replace([11,13,211,321,2212],[0,1,2,3,4])


target = target[0].tolist()
data = np.array(data)
target = np.array(target)

# Flatten the arrays
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.10, random_state=42)

batch_size=254
# Initializing model
model = tf.keras.Sequential([
    #tf.keras.layers.Conv1D(32, 2, activation='relu', input_shape=(10,10)),
    tf.keras.layers.Conv2D(10, kernel_size=(1,1), input_shape=(10,10,1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, kernel_size=(2,2), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(5, activation='softmax')
])
model.summary()
# Compiling model
model.compile(optimizer=tf.keras.optimizers.Adadelta(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


#### ONLY IF WE APPLY CONVOLUTIONAL NN
X_train = X_train.reshape((-1, 10,10,1))
X_test = X_test.reshape((-1, 10,10,1))

X_train = X_train/255
X_test = X_test/255




# Fitting model
model.fit(X_train, y_train, epochs=10, batch_size = batch_size, verbose=2, validation_data=(X_test, y_test))

# Apply the results to the test:
pkl_file = open(os.path.join('data\\test', 'data_test_file.pkl'), 'rb')
event_test = pickle.load(pkl_file)
data_test = pd.DataFrame(event_test)
data_test = pd.DataFrame(data_test[1])
data_test = data_test[1].tolist()
data_test = np.array(data_test)



data_test = data_test.reshape(-1, 10,10,1)

data_test = data_test/255

results_test = model.predict(data_test)

# Rename the columns and the index
col_names={0: "electron", 1 : "muon", 2:"pion", 3:"kaon", 4 : "proton"}
results_test = pd.DataFrame(results_test)
results_test.columns = results_test.columns.map(col_names)
results_test.index = results_test.index.set_names(['image'])

# WRITE CSV FILE
results_test.to_csv('submission.csv')


