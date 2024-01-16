import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score



print('Reading dataset from ../dataset/booking_train.csv')
df =  pd.read_csv('../dataset/booking_train.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')
df['type_of_meal']=df['type_of_meal'].str.lower().str.replace(' ', '_')
df['room_type']=df['room_type'].str.lower().str.replace(' ', '_')
df['market_segment_type']=df['market_segment_type'].str.lower().str.replace(' ', '_')
df = df[~df["date_of_reservation"].str.contains("-")]
df['month_of_reservation'] = pd.to_datetime(df['date_of_reservation'],format='%m/%d/%Y').dt.strftime('%b') 
df['booking_status'] = (df.booking_status == 'Canceled').astype(int)


print('Performing data split for train-val-test')
#Perform the train/validation/test split with Scikit-Learn
df_full_train, df_test = train_test_split(df,test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train,test_size=0.25, random_state=1)
df_full_train  = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_full_train = df_full_train['booking_status'].values
y_train = df_train['booking_status'].values
y_val = df_val['booking_status'].values
y_test = df_test['booking_status'].values
del df_train['booking_status']
del df_val['booking_status']
del df_test['booking_status']

print(f'Split completed successfully - train ->{len(df_train)}, val ->  {len(df_val)}, test -> {len(df_test)}')


print('Building Feature matrix')

numerical = ['number_of_adults','number_of_children','number_of_weekend_nights','number_of_week_nights','lead_time','p-c','p-not-c','average_price','special_requests']
categorical = ['type_of_meal','room_type','market_segment_type','car_parking_space','repeated','month_of_reservation']

full_train_dict = df_full_train[categorical + numerical].to_dict(orient='records')
train_dict = df_train[categorical + numerical].to_dict(orient='records')
val_dict = df_val[categorical + numerical].to_dict(orient='records')
test_dict = df_test[categorical + numerical].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dict)
X_val = dv.transform(val_dict)
X_test = dv.transform(test_dict)
X_full_train = dv.transform(full_train_dict)

features = list(dv.get_feature_names_out())

print('Feature matrix completed successfully with %d features' % len(features))


print('Building Xgb boost wrappers')
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)
dtrain_full = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)
print('Xgb boost wrappers completed successfully')


#validation
print('validation started')
xgb_params = {
    'eta': 0.1,
    'max_depth': 10,
    'min_child_weight': 2,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1
}

model = xgb.train(xgb_params, dtrain,
                  num_boost_round=290)

y_val_pred = model.predict(dval)
auc = roc_auc_score(y_val, y_val_pred)
#accuracy = accuracy_score(y_test, y_val_pred>0.5)
#precision = precision_score(y_test, y_val_pred>0.5)
#recall = recall_score(y_test, y_val_pred>0.5)
#f1_score = 2 * (precision * recall) / (precision + recall)

#print('validation dataset scores: auc:%2f - accuracy:%2f - precision:%2f - recall:%2f - f1_score:%2f' % (auc,accuracy,precision,recall,f1_score))
print('validation dataset scores: auc:%2f' % (auc))

#training final model.

print('training final model with eta=%f, max_depth=%d, min_child_depth=%d, num_boost_round=%d.' % (0.1,10,2,290))
xgb_params = {
    'eta': 0.1,
    'max_depth': 10,
    'min_child_weight': 2,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1
}

model = xgb.train(xgb_params, dtrain_full,
                  num_boost_round=290)

print('training completed successfully')


y_test_pred = model.predict(dtest)
auc = roc_auc_score(y_test, y_test_pred)
#accuracy = accuracy_score(y_test, y_test_pred>0.5)
#precision = precision_score(y_test, y_test_pred>0.5)
#recall = recall_score(y_test, y_test_pred>0.5)
#f1_score = 2 * (precision * recall) / (precision + recall)

#print('Predicting with test data complete with auc score %3f' % auc)
#print('test dataset scores: auc:%2f - accuracy:%2f - precision:%2f - recall:%2f - f1_score:%2f' % (auc,accuracy,precision,recall,f1_score))
print('test dataset scores: auc:%2f' % (auc))

with open('../model/cancellation-pred-model-xgb.bin', 'wb') as f_out:
    pickle.dump((dv,model),f_out)

print('Saved xgb model at : ../model/%s' % 'cancellation-pred-model-xgb.bin')    
