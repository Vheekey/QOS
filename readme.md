# QOS
AI model trained in predicting Quality of service

## To Start
1. Make sure Python is installed on your machine. You can check this by running:
```
python3 --version
```
If it's not installed, you can download it from [here](https://www.python.org/downloads/).

or

```
python --version
```

1. Create a virtual environment and activate it:
```
python3 -m venv venv
source venv/bin/activate
```

1. Install the required packages:
```
pip install -r requirements.txt
```

## To train
python3 optimize_qos_train_model.py <network_data.csv> <user_feedback.csv>

## To run the model:
python3 predict_qoe_from_model.py <model_file> <input_csv>
e.g python3 predict_qoe_from_model.py trained_model.joblib cobranet.csv


Sample data for testing:

```
bandwidth,latency,packet_loss,jitter
300,20,0.01,1
150,30,0.02,2
200,25,0.03,3
250,22,0.01,2
180,35,0.02,3
180,25,0.01,2
150,30,0.02,2]
```