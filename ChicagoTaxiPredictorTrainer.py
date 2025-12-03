import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")
chicago_taxi_dataset = chicago_taxi_dataset.dropna()
chicago_taxi_dataset.drop_duplicates(inplace=True)
milefeatures = chicago_taxi_dataset["TRIP_MILES"].values
labels = chicago_taxi_dataset["FARE"].values
epoch = 200000
milew = 0.0
mileb = 0.0
lr = 0.0001
n = len(chicago_taxi_dataset)

for i in range(epoch):
    milepred = milew * milefeatures + mileb
    miledw = (-2/n) * np.sum(milefeatures * (labels - milepred))
    miledb = (-2/n) * np.sum(labels - milepred)
    milew -= lr * miledw
    mileb -= lr * miledb

with open("wandb.txt", "w") as f:
    f.write(f"{round(milew, 8)}\n")
    f.write(f"{round(mileb, 8)}\n")
plt.scatter(milefeatures, labels, alpha=0.3, label="Actual Fares")
x_vals = np.linspace(min(milefeatures), max(milefeatures), 100)
y_vals = milew * x_vals + mileb
plt.plot(x_vals, y_vals, color="red", label="Regression Line")
plt.xlabel("Trip Miles")
plt.ylabel("Fare")
plt.title("Fare Predictions")
plt.legend()

plt.show()
