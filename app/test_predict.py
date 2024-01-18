import requests

server_url = 'http://localhost:9696/predict'

data = {
    "booking_id": "531a9073-3ad7-4f0d-8733-9c24718d228d",
    "type_of_meal": "meal_plan_2",
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
    "special_requests": 0
 }

result = requests.post(server_url, json=data).json()

print(result)
'''
Data whose booking was cancelled

{
    "booking_id": "531a9073-3ad7-4f0d-8733-9c24718d228d",
    "type_of_meal": "meal_plan_2",
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
    "special_requests": 0
 }
'''

'''
Data whose booking was not cancelled
        {
          "booking_id": "822a9073-3ad7-4f0d-8733-9c24718d228d",
           "type_of_meal": "meal_plan_1",
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
           "special_requests": 0
        }
'''


