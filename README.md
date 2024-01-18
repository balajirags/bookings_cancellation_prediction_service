# Predicting Hotel Booking Cancellations #

## Problem statement ##
In the dynamic hospitality industry, hotel bookings are subject to fluctuations in demand, and cancellations can significantly impact revenue and resource management. The objective of this capstone project is to develop a predictive model that accurately estimates the likelihood of a booking being canceled after it has been made.

### Goals  ###
1. Design and implement a machine learning model that predicts the probability of a hotel booking being canceled.
2. Utilize historical booking data, customer profiles, booking details, and other relevant features to train the model.
3. Evaluate the model's performance using appropriate metrics, such as auc,accuracy, precision, recall, and F1-score.
4. Provide actionable insights to hotel management for proactive decision-making and resource allocation.


Kaggle data set used - https://www.kaggle.com/datasets/youssefaboelwafa/hotel-booking-cancellation-prediction

## Solution ##
As part of addressing this challenge, firstly the dataset was cleaned up and trasformed as required. Then Exploratory Data Analysis(EDA) was performed to understand the feature importance. Metrics like `cancellation rate`,`risk ratio`,`mutual information`, `correlations` were analysed on the dataset. Post that several binary classification models were trained on hyper parameters and evaluated against auc, accuracy, precision, recall, f1. Here is a summary of the models and their respective auc scores.

**Note**: 
1. Data cleanup and EDA are done as a part of the first model training notebook - `./trainer/notebooks/logistic_regression_trainer.ipynb`. 
2. For each model training seperate notebooks have been created as below.
3. Checkout the './trainer/notebooks/compare_models.ipynb' to see a comparison between different models.

notebook |Model  | auc score
---------|------- | -------------
logistic_regression_trainer.ipynb | LogisticRegression | 86%
decision_tree_trainer.ipynb| DecisionTree | 92%
random_forest_trainer.ipynb| RandomForest | 93%
xg_boost_trainer.ipynb | XGBoost |   95%

Among these models, the 'XGBoost' model demonstrated the highest AUC score, signifying its superior predictive performance. Subsequently, this top-performing 'XGBoost' model was encapsulated within a Flask application and deployed within a local or cloud Kubernetes cluster. The Flask application exposes a '/predict' API endpoint, enabling users to obtain predictions for customer bookings efficiently. This deployment ensures the model's availability for real-time predictions and facilitates seamless integration into hotel booking systems.


## Note ##
 All trained models are checked into the repo under './model' directory.


### Project structure ###

Folder  | Description
------------- | -------------
dataset  | Directory containing training|test dataset
app  | Directory containing flask application which wraps the XGBoost model and provides inference
model    | Directory containing '.bin' model which was trianed.
trainer | Directory containing notebooks and script which were used for training the model. 
kube-config | Configuration related to kubernetes deployments.


## Pre-requisties ##
* python 3.10 or above.
* docker, Kind, kubectl
* pip3, pipenv  

## Installing dependencies ##
Use `pipenv install` to install dependencies from respective directories, Only if you want to train model and build images yourself.

* Folder `app` - contains dependencies related to flask and XGBoost. 
* Folder `trainer` - contains dependencies required for training the model.


## How to run app locally [without Docker] ##
1.  Clone this repo
2.  `cd repo/app`
3. `pipenv install`
4. `pipenv shell`
5. `python3 predict.py`
6. Test the application with postman or python script or shell script as given below.

## How to run app locally with Docker ##
1.  Clone this repo
2.  `cd repo`
3. `docker build -t cancellation-prediction:v1  -f  ./app/cancellation_prediction_app_image.dockerfile .`
4. `docker run -it --rm -p 9696:9696 cancellation-prediction:v1`
5.  check if the containers is up `docker ps`
6. Test the application with postman or python script.


## How to run on local kubernetes ##
1. Ensure [`kind`](https://kind.sigs.k8s.io/) kubernetes in installed
2. run `./deploy-local-kube.sh <cluster-name>`
3. `kubectl port-forward services/sports-gateway-service 9696:9696`
4. Test the application with postman or python script.
5. Once testing is completed delete your local cluster using `./delete-local-kube.sh`

## Testing with python script ##
1. `cd` into `./app` directory
2. `pipenv install`
3. `pipenv shell`
4. `python3 test_predict.py`

## Testing with curl ##
run `./test_predict.sh`
 
 or

```shell
  curl --request POST \
  --url http://localhost:9696/predict \
  --header 'Content-Type: application/json' \
  --data '{"type_of_meal": "meal_plan_1",
           "room_type": "room_type_1",
           "market_segment_type": "offline",
           "car_parking_space": 0,
           "repeated": 0,
           "month_of_reservation": "Oct",
           "number_of_adults": 2,
           "number_of_children": 0,
           "number_of_weekend_nights": 0,
           "number_of_week_nights": 3,
           "lead_time": 105,
           "p-c": 0,
           "p-not-c": 0,
           "average_price": 75.0,
           "special_requests": 0}'
  ```

### Sample data for testing ###
Data whose booking cancellation is likely to happen in the future(prediction=True)
```Shell
{"type_of_meal": "meal_plan_2",
 "room_type": "room_type_1",
 "market_segment_type": "offline",
 "car_parking_space": 0,
 "repeated": 0,
 "month_of_reservation": "Apr",
 "number_of_adults": 1,
 "number_of_children": 0,
 "number_of_weekend_nights": 2,
 "number_of_week_nights": 1,
 "lead_time": 80,
 "p-c": 0,
 "p-not-c": 0,
 "average_price": 100.0,
 "special_requests": 0}
```

Data whose booking is unlikely to be cancelled in the future(prediction=False)
```Shell
{"type_of_meal": "meal_plan_1",
           "room_type": "room_type_1",
           "market_segment_type": "offline",
           "car_parking_space": 0,
           "repeated": 0,
           "month_of_reservation": "Oct",
           "number_of_adults": 2,
           "number_of_children": 0,
           "number_of_weekend_nights": 0,
           "number_of_week_nights": 3,
           "lead_time": 105,
           "p-c": 0,
           "p-not-c": 0,
           "average_price": 75.0,
           "special_requests": 0}
```

## AWS Cloud Deployment ##

The app was deployed on AWS EKS cluster and tested. You can find the video of testing here. Due to cost reason, removed the cluster post testing.



## Service API ##

API  | Description | Response | Response-type
------------- | ------------- | -------------  | -------------
`/ping` | ping api to check the status | pong | String
`/predict`| Return the cancellation prediction for a booking|  check below sample | Json


Sample Response for '/predict'
```JSON
  {
	"cancellation_probability": 0.26556459069252014,
	"message": "unlikely to cancel the booking",
	"prediction": false
  }
 
```

Response
 Attributes  | Description | Response | Response-type
------------- | ------------- | -------------  | -------------
cancellation_probability | Probabilty of cancellation | 0 to 1 | float
prediction |   False - unlikely to cancel, True  - likely to cancel | true or false | boolean
message | Human readable message of the response |String |String
               


## Training the model from scratch ##
The trained model is already checked into the repos under `./model` directory along with the notebooks used for training and testing(`./trainer/notebooks`). Incase you want to do from scratch follow below steps

1. `cd into trainer` directory
2. `pipenv install`
3. `pipenv shell`
4. `python3 <model>.py` - Running this code saves the `.bin` model into `./model` directory

trin_scripts  | Description
-------------- | ---------------
logistic_regression_trainer | Logistic regression model for training,testing & saving model
decision_tree_trainer | Decision tree model for training,testing & saving model
random_forest_trainer | Random forest model for training,testing & saving model
xgb_trainer | xgboost model for training,testing & saving model

**EDA are done in notebooks.

5. you can test the model you have trained using `python3 test_predict.py`(modify the model as required in the test file)
