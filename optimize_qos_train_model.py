import sys
import subprocess
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump

# Function to install a package
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check and install required libraries
try:
    import pandas as pd
except ImportError:
    install("pandas")
    import pandas as pd

try:
    import numpy as np
except ImportError:
    install("numpy")
    import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    install("matplotlib")
    import matplotlib.pyplot as plt

try:
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
except ImportError:
    install("scikit-learn")
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

def load_data(network_file, feedback_file):
    try:
        network_data = pd.read_csv(network_file)
        user_feedback = pd.read_csv(feedback_file)
    except pd.errors.EmptyDataError:
        print("Error: One of the input files is empty or improperly formatted.")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: One of the input files was not found.")
        sys.exit(1)
    return network_data, user_feedback

def preprocess_data(network_data, user_feedback):
    # Example of feature engineering: creating a new feature 'latency_ratio'
    network_data['latency_ratio'] = network_data['latency'] / network_data['throughput']
    network_data['jitter_per_latency'] = network_data['jitter'] / network_data['latency']
    return network_data, user_feedback

def train_model(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    model = RandomForestRegressor(random_state=42)
    cv = min(5, len(X_train))  # Use 5 folds or less if fewer samples are available
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)
    best_model = grid_search.best_estimator_
    return best_model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    return predictions

def visualize_results(y_test, predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.reset_index(drop=True), label='True QoE')
    plt.plot(predictions, label='Predicted QoE')
    plt.legend()
    plt.xlabel('Sample Index Data')
    plt.ylabel('QoE')
    plt.title('True QoE vs Predicted QoE')
    plt.savefig('qoe_comparison.png')
    plt.show()

def display_image(image_path):
    if sys.platform == "win32":
        os.startfile(image_path)
    elif sys.platform == "darwin":
        os.system(f"open {image_path}")
    else:
        os.system(f"xdg-open {image_path}")

def save_model(model, model_file):
    try:
        dump(model, model_file)
        print(f"Model saved successfully to {model_file}")
    except Exception as e:
        print(f"Error saving model: {e}")

def main(network_file, feedback_file, model_file='trained_model.joblib'):
    network_data, user_feedback = load_data(network_file, feedback_file)
    network_data, user_feedback = preprocess_data(network_data, user_feedback)

    X = network_data[['throughput', 'latency', 'packet_loss', 'jitter', 'latency_ratio', 'jitter_per_latency']]
    y = user_feedback['qoe']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_test:", y_test.shape)
    
    model = train_model(X_train, y_train)
    
    predictions = evaluate_model(model, X_test, y_test)
    
    print("Shape of predictions:", predictions.shape)
    print("y_test values:", y_test)
    print("Predictions:", predictions)
    
    # Save the trained model
    save_model(model, model_file)
    
    visualize_results(y_test, predictions)
    display_image('qoe_comparison.png')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 optimize_qos_train_model.py <network_data.csv> <user_feedback.csv>")
    else:
        network_file = sys.argv[1]
        feedback_file = sys.argv[2]
        main(network_file, feedback_file)
