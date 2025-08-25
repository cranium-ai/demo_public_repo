import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegressionRegressor
from sklearn.metrics import mean_squared_error, r2_score
import argparse


def train_model():
    # Load the dataset
    df = pd.read_csv("revenue.csv")

    # Display the first few rows of the dataset
    print(df.head())

    X = df[["feature1", "feature2", "feature3"]]  # Features
    y = df["revenue"]  # Target variable

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create a Linear Regression model
    model = LinearRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Print the first few predictions
    print("Predictions on test data:")
    print(y_pred[:5])

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, help="Path to the data file")
    args = parser.parse_args()

    model = train_model()
    new_data = pd.read_csv(args.data_file)
    prediction = model.predict(new_data)
    print(f"Prediction for new data: {prediction}")
