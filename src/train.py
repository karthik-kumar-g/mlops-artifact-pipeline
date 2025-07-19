import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from src.utils import load_config

def train_model():

    config = load_config("config/config.json")


    digits = load_digits()
    X, y = digits.data, digits.target


    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)


    model = LogisticRegression(
        C=config["C"],
        solver=config["solver"],
        max_iter=config["max_iter"]
    )
    model.fit(X_train, y_train)

    
    joblib.dump(model, "model_train.pkl")
    print("Model saved as model_train.pkl")

if __name__ == "__main__":
    train_model()
