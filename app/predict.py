from flask import Flask
from flask import request
from flask import jsonify
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

model_path = os.getenv('MODEL_PATH', '../model/')
dv, model = load_model(f'{model_path}/cancellation-pred-model-xgb.bin')

prediction_message = {1:'likely to cancel the booking',
                    0:'unlikely to cancel the booking'}
                    
@app.route('/predict', methods=['POST'])
def predict_cancellation():
    try:
        booking_details = request.get_json()
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
    #response = predict(url)
    #print(response)
    app.run(debug=True, host='0.0.0.0', port=9696)