import pandas as pd

csv_path = "training.1600000.processed.noemoticon.csv"
df_full = pd.read_csv(csv_path, encoding='latin-1', header=None)

df_full.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

df_sampled = df_full.sample(n=20000, random_state=42)

output_path = "sentiment140_sampled_20000.csv"
df_sampled.to_csv(output_path, index=False)