import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
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

def train(df, y):
    data = df[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    dv.fit(data)
    X = dv.transform(data)
    model = RandomForestClassifier(n_estimators=30, max_depth=20, min_samples_leaf=5, random_state=5)
    model.fit(X_train, y_train)
    return dv, model


def predict(df, dv, model):
    data = df[categorical + numerical].to_dict(orient='records')
    
    X = dv.transform(data)

    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


#validation
print('validation started')
dv, model = train(df_train, y_train)
y_val_pred = predict(df_val, dv, model)

auc = roc_auc_score(y_val, y_val_pred)
#accuracy = accuracy_score(y_test, y_val_pred>0.5)
#precision = precision_score(y_test, y_val_pred>0.5)
#recall = recall_score(y_test, y_val_pred>0.5)
#f1_score = 2 * (precision * recall) / (precision + recall)

#print('validation dataset scores: auc:%2f - accuracy:%2f - precision:%2f - recall:%2f - f1_score:%2f' % (auc,accuracy,precision,recall,f1_score))
print('validation dataset scores: auc:%2f' % (auc))

#training final model.
print('training final model with max_depth=%d, min_samples_leaf=%d, n_estimator=%d.' % (20,5,30))
dv, model = train(df_full_train, y_full_train)
print('training completed successfully')

y_test_pred = predict(df_test, dv, model)
auc = roc_auc_score(y_test, y_test_pred)
#accuracy = accuracy_score(y_test, y_test_pred>0.5)
#precision = precision_score(y_test, y_test_pred>0.5)
#recall = recall_score(y_test, y_test_pred>0.5)
#f1_score = 2 * (precision * recall) / (precision + recall)

#print('test dataset scores: auc:%2f - accuracy:%2f - precision:%2f - recall:%2f - f1_score:%2f' % (auc,accuracy,precision,recall,f1_score))
print('test dataset scores: auc:%2f' % (auc))
with open('../model/cancellation-pred-model-random-forest.bin', 'wb') as f_out:
    pickle.dump((dv,model),f_out)

print('Saved Random forest model at : ../model/%s' % 'cancellation-pred-model-random-forest.bin')    
