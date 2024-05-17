import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import argparse

def train_model():
    # Load the dataset
    df = pd.read_csv('letter_grades.csv')

    # Display the first few rows of the dataset
    print(df.head())

    # Assuming the dataset has columns 'feature1', 'feature2', ... 'featureN' and 'grade'
    # Replace 'feature1', 'feature2', ..., 'featureN' with the actual feature columns in your dataset
    features = ['feature1', 'feature2', 'feature3']  # Example feature columns
    X = df[features]  # Features
    y = df['grade']  # Target variable

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create a Random Forest Classifier model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return model, scaler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--new_data_file', type=str, help='Path to the new data file')
    args = parser.parse_args()

    # Train the model
    model, scaler = train_model()

    # Load the new data
    new_data = pd.read_csv(args.new_data_file)

    # Assuming the new data has the same feature columns as the training data
    features = ['feature1', 'feature2', 'feature3']
    X_new = new_data[features]

    # Standardize the new data using the same scaler
    X_new = scaler.transform(X_new)

    # Use the trained model to make predictions on the new data
    y_pred_new = model.predict(X_new)
    # Print the predictions
    print("Predictions for new data:")
    print(y_pred_new)
