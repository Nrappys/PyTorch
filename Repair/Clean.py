import pandas as pd

df = pd.read_csv('Repair\\raw_data.csv') 

df['Label'].value_counts()

invalid_labels = ['ten', '-1', '99', 'O', '?', -1, 99]
df = df[~df['Label'].isin(invalid_labels)]
print(len(df))

df['Label'].value_counts()

for i in range(784):
    col_name = str(i)

    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    
    if df[col_name].isnull().any():
        mean_value = int(df[col_name].mean())
        df[col_name].fillna(mean_value, inplace=True)
    
    df[col_name] = df[col_name].clip(0, 255)

null_count = df.isnull().sum().sum()
print(f"count null : {null_count}")

valid_labels = [str(i) for i in range(10)] + list(range(10))
invalid_label_count = df[~df['Label'].isin(valid_labels)].shape[0]
print(f"invalid_label : {invalid_label_count}")
print(f"Label : {df['Label'].unique()}")

pixel_cols = [str(i) for i in range(784)]
out_of_range = 0
for col in pixel_cols:
    out_of_range += ((df[col] < 0) | (df[col] > 255)).sum()
print(f"out of range : {out_of_range}")

non_numeric = 0
for col in pixel_cols:
    non_numeric += (~df[col].apply(lambda x: isinstance(x, (int, float)))).sum()
print(f"non numeric : {non_numeric}")

if null_count == 0 and invalid_label_count == 0 and out_of_range == 0 and non_numeric == 0:
    print("clean complete")
else:
    print("clean incomplete")

df.to_csv('Repair\\cleaned_data.csv', index=False)
print("saved cleaned data to cleaned_data.csv")