import pandas as pd
import joblib
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the model and preprocessing objects
best_model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

# Define features
categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
numerical_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
encoded_columns = encoder.get_feature_names_out(categorical_features)

# Create the GUI
def predict_price():
    try:
        # Retrieve user inputs
        area = float(area_entry.get())
        bedrooms = int(bedrooms_entry.get())
        bathrooms = int(bathrooms_entry.get())
        stories = int(stories_entry.get())
        mainroad = mainroad_var.get().lower()
        guestroom = guestroom_var.get().lower()
        basement = basement_var.get().lower()
        hotwaterheating = hotwaterheating_var.get().lower()
        airconditioning = airconditioning_var.get().lower()
        parking = int(parking_entry.get())
        prefarea = prefarea_var.get().lower()
        furnishingstatus = furnishingstatus_var.get().lower()

        # Create a DataFrame for the new data
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

        # Apply preprocessing
        new_data_encoded = encoder.transform(new_data[categorical_features])
        new_data_encoded_df = pd.DataFrame(new_data_encoded, columns=encoded_columns)
        new_data_final = pd.concat([new_data[numerical_features], new_data_encoded_df], axis=1)
        new_data_scaled = scaler.transform(new_data_final)

        # Predict
        prediction = best_model.predict(new_data_scaled)[0]

        # Show the prediction
        messagebox.showinfo("Prediction", f"Predicted House Price: ${prediction:,.2f}")

    except ValueError as e:
        messagebox.showerror("Input Error", f"Invalid input: {e}. Please enter valid data.")

# Create the main window
root = tk.Tk()
root.title("House Price Prediction")
root.geometry("700x650")  # Set the size of the window

# Set a font and color scheme
font_large = ('Arial', 14)
font_medium = ('Arial', 12)
font_small = ('Arial', 10)
bg_color = "#f9f9f9"
text_color = "#000000"  # Red color for text
button_color = "#4CAF50"
button_text_color = "#ffffff"

root.configure(bg=bg_color)

# Create and place labels and entry widgets
def create_label(root, text, row, column, colspan=1):
    label = tk.Label(root, text=text, bg=bg_color, font=font_medium, fg=text_color, anchor='w', padx=10)
    label.grid(row=row, column=column, columnspan=colspan, sticky='w', padx=10, pady=10)

def create_entry(root, row, column):
    entry = tk.Entry(root, font=font_medium, fg=text_color)  # Set text color to red
    entry.grid(row=row, column=column, padx=10, pady=10, sticky='ew')
    return entry

def create_radiobuttons(root, text, variable, values, row, column):
    tk.Label(root, text=text, bg=bg_color, font=font_medium, fg=text_color, anchor='w', padx=10).grid(row=row, column=column, sticky='w', padx=10, pady=10)
    for i, value in enumerate(values):
        tk.Radiobutton(root, text=value, variable=variable, value=value.lower(), font=font_small, bg=bg_color, fg=text_color).grid(row=row, column=column + i + 1, padx=5, pady=10, sticky='w')

# Input fields
create_label(root, "Area (sq ft):", 0, 0)
area_entry = create_entry(root, 0, 1)

create_label(root, "Bedrooms:", 1, 0)
bedrooms_entry = create_entry(root, 1, 1)

create_label(root, "Bathrooms:", 2, 0)
bathrooms_entry = create_entry(root, 2, 1)

create_label(root, "Stories:", 3, 0)
stories_entry = create_entry(root, 3, 1)

create_label(root, "Parking spaces:", 4, 0)
parking_entry = create_entry(root, 4, 1)

# Radio buttons for categorical features
mainroad_var = tk.StringVar(value='no')
create_radiobuttons(root, "Main road:", mainroad_var, ["Yes", "No"], 5, 0)

guestroom_var = tk.StringVar(value='no')
create_radiobuttons(root, "Guest room:", guestroom_var, ["Yes", "No"], 6, 0)

basement_var = tk.StringVar(value='no')
create_radiobuttons(root, "Basement:", basement_var, ["Yes", "No"], 7, 0)

hotwaterheating_var = tk.StringVar(value='no')
create_radiobuttons(root, "Hot water heating:", hotwaterheating_var, ["Yes", "No"], 8, 0)

airconditioning_var = tk.StringVar(value='no')
create_radiobuttons(root, "Air conditioning:", airconditioning_var, ["Yes", "No"], 9, 0)

prefarea_var = tk.StringVar(value='no')
create_radiobuttons(root, "Preferred area:", prefarea_var, ["Yes", "No"], 10, 0)

furnishingstatus_var = tk.StringVar(value='unfurnished')
create_radiobuttons(root, "Furnishing status:", furnishingstatus_var, ["Furnished", "Semi-furnished", "Unfurnished"], 11, 0)

# Create and place the Predict button
predict_button = tk.Button(root, text="Predict Price", command=predict_price, bg=button_color, fg=button_text_color, font=font_large)
predict_button.grid(row=12, columnspan=4, pady=20)
predict_button.config(height=0, width=15)


# Run the application
root.mainloop()
