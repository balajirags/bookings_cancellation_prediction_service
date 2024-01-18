from flask import Flask
from flask import request
from flask import jsonify
from flask import g
from flask_expects_json import expects_json
from urllib.error import HTTPError
import pickle
import xgboost as xgb
import os

app = Flask("cancellation-prediction")


def load_model(file_path):
  with open(file_path, 'rb') as f_in:
    dv, model = pickle.load(f_in)  
  return dv, model


def handle_error(e, code):
    return jsonify({
            'status': 'error',
            'message': f'{e}'
        }), code 

@app.errorhandler(400)    
def bad_request(error):
     return jsonify({
            'status': 'error',
            'message': f'{error}'
        }), 400 

    
schema = {
    'type': 'object',
    'properties': {
        'booking_id': {'type': 'string'},
        'type_of_meal': {'type': 'string'},
        'room_type': {'type': 'string'},
        'market_segment_type': {'type': 'string'},
        'car_parking_space': {'type': 'number'},
        'repeated': {'type': 'number'},
        'month_of_reservation': {'type': 'string'},
        'number_of_adults': {'type': 'number'},
        'number_of_children': {'type': 'number'},
        'number_of_weekend_nights': {'type': 'number'},
        'number_of_week_nights': {'type': 'number'},
        'lead_time': {'type': 'number'},
        'p-c': {'type': 'number'},
        'p-not-c': {'type': 'number'},
        'average_price': {'type': 'number'},
        'special_requests': {'type': 'number'}
    },
    'required': ['type_of_meal', 'room_type','market_segment_type','car_parking_space','repeated','month_of_reservation','number_of_adults','number_of_children',
                 'number_of_weekend_nights','number_of_week_nights','lead_time','p-c','p-not-c','average_price','special_requests']
}    


model_path = os.getenv('MODEL_PATH', '../model/')
dv, model = load_model(f'{model_path}/cancellation-pred-model-xgb.bin')

prediction_message = {1:'likely to cancel the booking',
                    0:'unlikely to cancel the booking'}
                    
@app.route('/predict', methods=['POST'])
@expects_json(schema)
def predict_cancellation():
    try:
        booking_details = g.data
        print(f'Received data {booking_details  }')
        X = dv.transform(booking_details)
        dtest = xgb.DMatrix(X,feature_names=list(dv.get_feature_names_out()))
        y_pred = model.predict(dtest)
        cancellation = y_pred[0] > 0.5
        return jsonify({
            'prediction': bool(cancellation),
            'cancellation_probability': float(y_pred[0]),
            'message':prediction_message[cancellation.astype(bool)]
            }), 200 
    except HTTPError as err:
        print(err.code) 
        return handle_error(err,err.code)
    except Exception as e:
        return handle_error(e,500)

@app.route('/ping', methods=['GET'])
def ping_api():
    return 'pong'        


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)