
# 🌦️ Weather Prediction Using KNN

This project is a machine learning-based weather prediction system using the **K-Nearest Neighbors (KNN)** algorithm. It predicts whether it will **rain (0)** or **not rain (1)** based on weather features in **Uhana, Ampara, Sri Lanka**.

---

## 📊 Dataset

A synthetic dataset of **1000 samples** with the following features:

| Feature       | Description                   |
|---------------|-------------------------------|
| Temperature   | Air temperature in °C         |
| Humidity      | Relative humidity in %        |
| WindSpeed     | Wind speed in km/h            |
| Pressure      | Atmospheric pressure in hPa   |

**Label:**
- `0`: Rain
- `1`: Not Rain

The dataset is saved as:  
📁 `weather_dataset_uhana.csv`

---

## 🚀 Technologies Used

- Python 🐍
- Jupyter Notebook
- Scikit-learn
- Pandas
- NumPy
- Matplotlib / Seaborn
- Joblib

---

## 🛠️ Project Structure

```
.
├── weather_dataset_uhana.csv         # Dataset
├── knn_weather_model.pkl             # Trained KNN model
├── scaler.pkl                        # StandardScaler used for preprocessing
├── train_model.ipynb                 # Notebook to train and save model
├── predict_model.ipynb               # Notebook to load model and predict
└── README.md                         # This file
```

---

## 🧪 How to Use

### 1️⃣ Train the Model
Open `train_model.ipynb` and run all cells to:
- Load and explore dataset
- Preprocess the data
- Train a KNN model
- Save the model and scaler to `.pkl` files

### 2️⃣ Predict with Saved Model
Open `predict_model.ipynb` and:
- Load the model and scaler
- Input new weather data (temperature, humidity, wind speed, pressure)
- Get a prediction: `0` (Rain) or `1` (Not Rain)

Example:
```python
import pandas as pd

new_input = pd.DataFrame([[27.0, 85.0, 10.0, 1005.0]], 
                         columns=['Temperature', 'Humidity', 'WindSpeed', 'Pressure'])

scaled_input = scaler.transform(new_input)
prediction = model.predict(scaled_input)
print("Prediction (0 = Rain, 1 = Not Rain):", prediction[0])
```

---

## 💡 Notes

- The dataset is **synthetic** but based on realistic weather conditions for Uhana.
- Feature scaling is **required** before using KNN.
- Avoid the warning: "X does not have valid feature names..." by using a DataFrame for prediction input.

---

## 👤 Author

**Isuru Senarath**  
📍 Uhana, Ampara, Sri Lanka  
💬 Feel free to modify and extend this project as needed!

---

## 📄 License

This project is free to use and modify for educational and non-commercial purposes.
