import pickle
import xgboost as xgb

def load_model(file_path):
  with open(file_path, 'rb') as f_in:
    dv, model = pickle.load(f_in)  
  return dv, model


test_booking = {'type_of_meal': 'meal_plan_1',
 'room_type': 'room_type_1',
 'market_segment_type': 'offline',
 'car_parking_space': 0,
 'repeated': 0,
 'month_of_reservation': 'Oct',
 'number_of_adults': 2,
 'number_of_children': 0,
 'number_of_weekend_nights': 0,
 'number_of_week_nights': 3,
 'lead_time': 105,
 'p-c': 0,
 'p-not-c': 0,
 'average_price': 75.0,
 'special_requests': 0}



dv, model = load_model('../model/cancellation-pred-model-xgb.bin')
X = dv.transform(test_booking)
dtest = xgb.DMatrix(X,feature_names=list(dv.get_feature_names_out()))
y_pred = model.predict(dtest)
y_pred[0]
print('Probablity of cancellation(using XGB tree) - ', y_pred[0])