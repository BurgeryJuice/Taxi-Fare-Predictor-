import pandas as pd
import numpy as np

with open("wandb.txt", "r") as f:
    lines = f.readlines()
    weight = float(lines[0].strip())
    bias = float(lines[1].strip())

Dist = 6.1
predicted_fare = weight * Dist + bias
print("The predicted fare is around ", predicted_fare)