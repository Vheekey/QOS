# QOS
AI model trained in predicting Quality of service

To train
python3 optimize_qos_train_model.py <network_data.csv> <user_feedback.csv>

To run the model:
python3 predict_qoe_from_model.py <model_file> <input_csv>