import pandas as pd
df= pd.read_csv("data/raw/data.csv",header=None)
df.columns=["id","value"]
df ["value"]*=2
df.to_csv("data/processed/process.csv", index=False)
