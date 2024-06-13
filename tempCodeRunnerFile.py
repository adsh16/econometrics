import pandas as pd

# Load the first CSV containing your existing data
df_existing = pd.read_csv('output_data.csv')

# Load the CSV containing the Gini index data
df_gini = pd.read_csv('GiniUse.csv')

# Clean and normalize state and district names in both DataFrames
def normalize_name(name):
  # Remove leading/trailing whitespace and convert to lowercase
    return name.strip().lower()  

df_existing['state'] = df_existing['state'].apply(normalize_name)
df_existing['district'] = df_existing['district'].apply(normalize_name)
df_gini['Districts/State'] = df_gini['Districts/State'].apply(normalize_name)

# Merge the DataFrames based on matching state names
merged_df = pd.merge(df_existing, df_gini, left_on='state', right_on='Districts/State', how='left')

# Fill missing Gini index values using forward fill
merged_df['Gini'].ffill(inplace=True)

# Drop the redundant 'Districts/State' column
merged_df.drop('Districts/State', axis=1, inplace=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('output_data_with_gini.csv', index=False)
