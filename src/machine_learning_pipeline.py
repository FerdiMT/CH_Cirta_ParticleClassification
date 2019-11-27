import numpy as np
import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb

directory= 'data\\train'

#Loop over all the pkl files in the directory, open them and append all of them into a dataframe made of arrays.
data = pd.DataFrame()
target = pd.DataFrame()

### LOOP IN ORDER NOT TO HAVE SO MANY FILES

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

# Target labels replace the numbers for 0-4:
dic_types={11: 0, 13 : 1, 211:2, 321:3, 2212 : 4}
target = target.replace(dic_types)
# Reset indexes
data.reset_index(drop=True, inplace=True)
target.reset_index(drop=True, inplace=True)
print(target[0].value_counts())


# Udersampling the largest classes
undersampling_dictionary = {
    0:3138,
    1:1237,
    2:20000,
    3:20000,
    4:20000
}
rus = RandomUnderSampler(random_state=33, sampling_strategy=undersampling_dictionary)
data, target = rus.fit_resample(data, target)
data = pd.DataFrame(data)
target = pd.DataFrame(target)

# OVERSAMPLING MINORITY CLASSES
smote_dictionary = {
    0:20000,
    1:20000,
    2:20000,
    3:20000,
    4:20000
}
sm = SMOTE(random_state=33, sampling_strategy=smote_dictionary)
data, target = sm.fit_resample(data, target)
data = pd.DataFrame(data)
target = pd.DataFrame(target)
print(target[0].value_counts())


# Split train and test data.
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.20, random_state=42)

# Create pipeline
pipeline_xgb = Pipeline(
    [('feature_reduction', PCA(n_components=10)),
     ('xgb', xgb.XGBClassifier())
])

params={}
#params['feature_reduction__n_components'] = [5,10]
params['xgb__learning_rate'] =  [0.05]
params['xgb__objective']= ['multi:softprob']
params['xgb__max_depth'] = [4, 5, 6]

CV = GridSearchCV(pipeline_xgb, params, scoring = 'neg_log_loss', n_jobs= 1)
CV.fit(X_train, y_train)
print(CV.best_params_)
print(CV.best_score_)

y_pred = CV.predict_proba(X_test)

# Predict on the test set.
pkl_file = open(os.path.join('data\\test', 'data_test_file.pkl'), 'rb')
event_test = pickle.load(pkl_file)
data_test = pd.DataFrame(event_test)
data_test = pd.DataFrame(data_test[1])
data_test = data_test[1].tolist()
data_test = np.array(data_test)
data_test = [np.concatenate((i)) for i in data_test]
data_test = pd.DataFrame(data_test)
results_test = CV.predict_proba(data_test)
col_names={0: "electron", 1 : "muon", 2:"pion", 3:"kaon", 4 : "proton"}
results_test = pd.DataFrame(results_test)
results_test.columns = results_test.columns.map(col_names)
results_test.index = results_test.index.set_names(['image'])

# WRITE CSV FILE
results_test.to_csv('submission_ML_pipeline.csv')




