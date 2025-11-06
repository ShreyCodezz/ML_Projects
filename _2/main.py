from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

d = {
    "screen_time":      [2, 4, 5, 6, 7, 8, 3, 4, 2, 1, 9, 10, 5, 6, 7],
    "caffeine_intake":  [1, 2, 1, 3, 2, 3, 1, 2, 1, 0, 4, 5, 2, 3, 3],
    "exercise_hours":   [2, 1, 0, 1, 0, 0, 2, 3, 2, 3, 0, 0, 1, 2, 1],
    "stress_level":     [3, 4, 5, 6, 7, 8, 2, 3, 4, 2, 9, 10, 5, 6, 7],
    "sleep_hours":      [8, 7, 6, 5, 4, 3, 8, 9, 7, 9, 4, 3, 6, 5, 4],
    "sleep_quality":    [9, 8, 7, 6, 5, 3, 9, 10, 8, 10, 4, 2, 6, 5, 4]
}


df = pd.DataFrame(d)

X = df[["screen_time", "caffeine_intake", "exercise_hours", "stress_level", "sleep_hours"]]
y = df["sleep_quality"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", round(mse, 2))

print("\nTry your own values:")
sct = float(input("Screen time (hours): "))
cft = float(input("Caffeine intake (cups): "))
exh = float(input("Exercise hours: "))
stl = float(input("Stress level (1–10): "))
slh = float(input("Sleep hours: "))

user_data = pd.DataFrame([[sct, cft, exh, stl, slh]],
    columns=["screen_time", "caffeine_intake", "exercise_hours", "stress_level", "sleep_hours"])
user_scaled = scaler.transform(user_data)

your_pred = model.predict(user_scaled)
score = max(0, min(10, your_pred[0]))

print(f"\nPredicted Sleep Quality Score: {score:.2f}/10")

plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Sleep Quality")
plt.ylabel("Predicted Sleep Quality")
plt.title("Sleep Quality Prediction Graph")
plt.show()

