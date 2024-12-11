#importing all the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import numpy as np

# Function to predict weather and Visualize the data
def predict_weather():
    try:
        file_path = "weather.csv"  # My data file
        data = pd.read_csv(file_path)

        if 'Date' not in data.columns or 'Temperature' not in data.columns:
            messagebox.showerror("Error", "File must contain 'Date' and 'Temperature' columns")
            return

        # Converting  Date to datetime and extract features
        data['Date'] = pd.to_datetime(data['Date'])
        data['DayOfYear'] = data['Date'].dt.dayofyear
        data['Year'] = data['Date'].dt.year

        # Preparing data for training
        X = data[['DayOfYear', 'Year']]  # Features
        y = data['Temperature']  # Target

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        print(f"Cross-Validation MSE: {np.mean(cv_scores):.2f}")

        # Predict for next 7 days
        last_date = data['Date'].max()
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
        future_data = pd.DataFrame({
            'DayOfYear': [date.dayofyear for date in future_dates],
            'Year': [date.year for date in future_dates]
        })
        predictions = model.predict(future_data)

        # Visualization 1: Historical Data
        plt.figure(figsize=(10, 6))
        plt.plot(data['Date'], y, label="Historical Data", color='blue')
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        plt.title("Historical Weather Data")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Visualization 2: Predicted Data
        plt.figure(figsize=(10, 6))
        plt.plot(future_dates, predictions, label="Predicted Temperatures", color='orange', linestyle='--', marker='o')
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        plt.title("Predicted Temperatures for Next 7 Days")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Display predictions in a new window
        result_window = tk.Toplevel()
        result_window.title("Predicted Temperatures")
        result_window.geometry("400x300")
        result_window.configure(bg="#f2f4f8")

        result_label = ttk.Label(result_window, text="Predicted Temperatures for Next 7 Days:", style="TLabel")
        result_label.pack(pady=10)

        for i, (date, temp) in enumerate(zip(future_dates, predictions), start=1):
            day_label = ttk.Label(result_window, text=f"Day {i} ({date.strftime('%Y-%m-%d')}): {temp:.2f} °C",
                                  style="TLabel")
            day_label.pack()

        # Calculate accuracy on test set
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        accuracy_label = ttk.Label(result_window, text=f"Mean Squared Error on Test Set: {mse:.2f}", style="TLabel")
        accuracy_label.pack(pady=10)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Tkinter GUI
app = tk.Tk()
app.title("Weather Prediction Tool")
app.geometry("500x300")
app.configure(bg="#f2f4f8")

# Fonts and Colors
header_font = ("Helvetica", 18, "bold")
subheader_font = ("Helvetica", 14)
bg_color = "#f2f4f8"
btn_color = "#4caf50"
btn_text_color = "white"

# Header
header_label = tk.Label(app, text="Weather Prediction Tool", font=header_font, bg=bg_color, fg="#333")
header_label.pack(pady=20)

# Subheader
subheader_label = tk.Label(app, text="Using 'weather.csv' file to predict next 7 days' temperatures",
                           font=subheader_font, bg=bg_color, fg="#666")
subheader_label.pack(pady=10)

# Predict Button
predict_button = tk.Button(app, text="Predict and Visualize Weather", command=predict_weather,
                           font=("Helvetica", 12, "bold"),
                           bg=btn_color, fg=btn_text_color, padx=10, pady=5, relief="flat")
predict_button.pack(pady=20)

# Footer
footer_label = tk.Label(app, text="Ensure 'weather.csv' is in the same directory as this script",
                        font=("Helvetica", 10), bg=bg_color, fg="#999")
footer_label.pack(side="bottom", pady=10)

app.mainloop()