import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import joblib


# Function to load the trained model
def load_model(model_file):
    try:
        model = joblib.load(model_file)
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None


# Function to estimate throughput based on bandwidth and packet loss
def estimate_throughput(bandwidth, packet_loss):
    return bandwidth * (1 - packet_loss)


# Function to predict QoE for a single set of parameters
def predict_qoe(model, bandwidth, latency, packet_loss, jitter):
    try:
        # Estimate throughput using bandwidth and packet loss
        estimated_throughput = estimate_throughput(bandwidth, packet_loss)

        # Prepare the input data for prediction
        data = pd.DataFrame({
            'estimated_throughput': [estimated_throughput],
            'latency': [latency],
            'packet_loss': [packet_loss],
            'jitter': [jitter],
            'latency_ratio': [latency / estimated_throughput],  # Feature engineering
            'jitter_per_latency': [jitter / latency]  # Feature engineering
        })

        # Perform prediction
        prediction = model.predict(data)
        return prediction[0]
    except Exception as e:
        print(f"Error predicting QoE: {e}")
        return None


# Function to predict QoE for multiple sets of parameters from a CSV file
def predict_qoe_from_csv(model, csv_file):
    try:
        # Read the CSV file
        data = pd.read_csv(csv_file)

        # Predict QoE for each row in the CSV file
        predictionsArray = []
        for index, row in data.iterrows():
            bandwidth = row['bandwidth']
            latency = row['latency']
            packet_loss = row['packet_loss']
            jitter = row['jitter']

            print(f"Predicting for sample {index + 1}: bandwidth={bandwidth}, latency={latency}, packet_loss={packet_loss}, jitter={jitter}")

            prediction = predict_qoe(model, bandwidth, latency, packet_loss, jitter)

            print(f"Prediction: {prediction}")  # Check if prediction is made
            if prediction is not None:
                predictionsArray.append(prediction)
                print(
                    f"Sample {index + 1}: bandwidth={bandwidth}, latency={latency}, packet_loss={packet_loss}, jitter={jitter} => QoE={prediction:.2f}")
            else:
                predictionsArray.append(np.nan)
                print(
                    f"Sample {index + 1}: bandwidth={bandwidth}, latency={latency}, packet_loss={packet_loss}, jitter={jitter} => Failed to predict QoE")

        # Plotting the predictions
        plt.figure(figsize=(10, 5))
        plt.plot(predictionsArray, marker='o', linestyle='-', color='b', label='Predicted QoE')
        plt.xlabel('Sample Data')
        plt.ylabel('QoE')
        plt.title('Predicted QoE from CSV')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return predictionsArray
    except Exception as e:
        print(f"Error predicting QoE from CSV: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 predict_qoe_from_model.py <model_file> <input_csv>")
        sys.exit(1)

    model_file = sys.argv[1]
    input_csv = sys.argv[2]

    # Load the trained model
    model = load_model(model_file)
    if model is None:
        print("Failed to load the model. Exiting.")
        sys.exit(1)

    # Predict QoE from CSV
    predictions = predict_qoe_from_csv(model, input_csv)
    if predictions is not None:
        print("\nAll predictions:")
        for i, pred in enumerate(predictions):
            print(f"Sample {i + 1}: {pred:.2f}")
    else:
        print("Failed to make predictions.")
