import joblib
from src.utils import load_data

def main():
    X, y = load_data()

    # Load trained model
    model = joblib.load("model_train.pkl")

    # Make predictions
    predictions = model.predict(X)

    # Display predictions
    print("Sample predictions:", predictions[:10])

if __name__ == "__main__":
    main()
