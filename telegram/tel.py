import pandas as pd

p = pd.read_csv("tg.csv", index_col=0)
print(p)
print(p.iloc[0])
print(p.iloc[0][2])
p.iat[0, 3] = "cat"
p.to_csv("tg.csv", columns=p.columns, index=False)
