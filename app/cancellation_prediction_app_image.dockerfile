FROM python:3.10-slim

RUN pip install pipenv

ENV MODEL_PATH "./"
# Set the working directory in the container
WORKDIR /app

COPY ["./app/Pipfile", "./app/Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["./app/predict.py", "./"]
COPY ["./model/cancellation-pred-model-xgb.bin", "./"]

EXPOSE 9696

CMD ["python", "predict.py", "0.0.0.0:9696"]