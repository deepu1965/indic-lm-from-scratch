import pandas as pd

df = pd.read_csv("nepali/outputs/logs/training_log.csv")

val = df[df["val_loss"].notna()]

best = val.loc[val["val_loss"].idxmin()]

print("Best validation result:")
print(best)

print("\nFinal validation result:")
print(val.iloc[-1])