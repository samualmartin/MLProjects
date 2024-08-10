
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('dataset/housing_data.csv')

# Define features and target
categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
numerical_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
X = df[numerical_features + categorical_features]
y = df['price']

# One-Hot Encoding
encoder = OneHotEncoder(drop='first', sparse=False)
X_encoded = encoder.fit_transform(X[categorical_features])
encoded_columns = encoder.get_feature_names_out(categorical_features)
X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_columns)
X_final = pd.concat([X[numerical_features], X_encoded_df], axis=1)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R^2': r2}
    print(f"{name} Mean Squared Error: {mse}")
    print(f"{name} R^2 Score: {r2}")

# Select the best model based on R^2 score
best_model_name = max(results, key=lambda x: results[x]['R^2'])
best_model = models[best_model_name]
print(f"Best model based on R^2 score: {best_model_name}")

# Save the best model and preprocessing objects
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')

# Load the model and preprocessing objects
best_model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

# Function to get user input
def get_user_input():
    print("Please enter the following details about the house:")
    area = float(input("Area (in sq ft): "))
    bedrooms = int(input("Number of bedrooms: "))
    bathrooms = int(input("Number of bathrooms: "))
    stories = int(input("Number of stories: "))
    mainroad = input("Main road (yes/no): ").lower()
    guestroom = input("Guest room (yes/no): ").lower()
    basement = input("Basement (yes/no): ").lower()
    hotwaterheating = input("Hot water heating (yes/no): ").lower()
    airconditioning = input("Air conditioning (yes/no): ").lower()
    parking = int(input("Parking spaces: "))
    prefarea = input("Preferred area (yes/no): ").lower()
    furnishingstatus = input("Furnishing status (furnished/semi-furnished/unfurnished): ").lower()

    new_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad': [mainroad],
        'guestroom': [guestroom],
        'basement': [basement],
        'hotwaterheating': [hotwaterheating],
        'airconditioning': [airconditioning],
        'parking': [parking],
        'prefarea': [prefarea],
        'furnishingstatus': [furnishingstatus]
    })

    return new_data

# Get user input
new_data = get_user_input()

# Apply preprocessing to new data
new_data_encoded = encoder.transform(new_data[categorical_features])
new_data_encoded_df = pd.DataFrame(new_data_encoded, columns=encoded_columns)
new_data_final = pd.concat([new_data[numerical_features], new_data_encoded_df], axis=1)
new_data_scaled = scaler.transform(new_data_final)

# Make prediction
predictions = best_model.predict(new_data_scaled)
print("Predicted Price:", predictions[0])
