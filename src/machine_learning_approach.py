import numpy as np
import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm

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
        data_ = [np.concatenate((i)) for i in data_]
        data_ = pd.DataFrame(data_)
        target_ = pd.DataFrame(target_)
        data = data.append(data_)
        target = target.append(target_)
        continue
    else:
        continue

# Feature creation: max value per observation
new_data = pd.DataFrame()
new_data['highest_value'] = data.max(axis=1)

# Standard scaler
from sklearn.preprocessing import StandardScaler
data = StandardScaler().fit_transform(data)

# PCA and save the results in the new data:
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
pca.fit(data)
data = pca.transform(data)
data = pd.DataFrame(data)


dic_types={11: 0, 13 : 1, 211:2, 321:3, 2212 : 4}
target = target.replace(dic_types)


#### Remove labels from the most represented category


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.20, random_state=42)

parameters = {
    'objective': 'multiclass',
    'num_class': 5,
    'metric': 'multi_logloss',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 1
}

train_data = lightgbm.Dataset(X_train, label=y_train)
validation_data = lightgbm.Dataset(X_test, label=y_test, reference=train_data)

model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=validation_data,
                       num_boost_round=800,
                       early_stopping_rounds=100)

pkl_file = open(os.path.join('data\\test', 'data_test_file.pkl'), 'rb')
event_test = pickle.load(pkl_file)
data_test = pd.DataFrame(event_test)
data_test = pd.DataFrame(data_test[1])
data_test = data_test[1].tolist()
data_test = np.array(data_test)
data_test = [np.concatenate((i)) for i in data_test]

data_test = StandardScaler().fit_transform(data_test)

pca.fit(data_test)
data_test = pca.transform(data_test)
data_test = pd.DataFrame(data_test)

results_test = model.predict(data_test)

col_names={0: "electron", 1 : "muon", 2:"pion", 3:"kaon", 4 : "proton"}

results_test = pd.DataFrame(results_test)
results_test.columns = results_test.columns.map(col_names)
results_test.index = results_test.index.set_names(['image'])

# WRITE CSV FILE
results_test.to_csv('submission_ML.csv')


print('caca')