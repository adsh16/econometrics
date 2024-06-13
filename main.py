## ---------------------------------------------------------------------------------
# Splliting into has_data and missing_data for residualsodiumcarbonate
## ---------------------------------------------------------------------------------
import pandas as pd

# Read the CSV file
df = pd.read_csv('District-Level_GWQ_AllYears.csv')

# Split data into two dataframes based on presence of 'residualsodiumcarbonate'
has_data_df = df[df['residualsodiumcarbonate'].notna()]
missing_data_df = df[df['residualsodiumcarbonate'].isna()]

# Define output file paths
has_data_file = 'has_data.csv'
missing_data_file = 'missing_data.csv'

# Write dataframes to CSV files
has_data_df.to_csv(has_data_file, index=False)
missing_data_df.to_csv(missing_data_file, index=False)

print(f"Entries with data saved to: {has_data_file}")
print(f"Entries without data saved to: {missing_data_file}")


## ---------------------------------------------------------------------------------
# Missing data analysis
## ---------------------------------------------------------------------------------
import pandas as pd
# Load the CSV file into a DataFrame
df = pd.read_csv('missing_data.csv')
# Group by state and year to count missing values
missing_counts = df.groupby(['state'])['residualsodiumcarbonate'].apply(lambda x: x.isnull().sum()).reset_index(name='missing_count')
# Filter out rows with missing counts (non-zero)
missing_counts = missing_counts[missing_counts['missing_count'] > 0]
# Sort the DataFrame by 'missing_count' in descending order
missing_counts_sorted = missing_counts.sort_values(by='missing_count', ascending=False)
# Display states with missing values and their counts (sorted)
print("States with Missing Values (Sorted by Missing Count):")
print(missing_counts_sorted)

'''
state  missing_count
0                    Andaman And Nicobar Islands             33
1                                 Andhra Pradesh             47
2                              Arunachal Pradesh             80
3                                          Assam            257
4                                          Bihar            421
5                                     Chandigarh              7
6                                   Chhattisgarh            144
7                                          Delhi             87
8                                            Goa             22
9                                        Gujarat             91
10                                       Haryana            114
11                              Himachal Pradesh             46
12                             Jammu And Kashmir            161
13                                     Jharkhand            310
14                                     Karnataka            457
15                                        Kerala            107
16                                Madhya Pradesh            222
17                                   Maharashtra            188
18                                     Meghalaya            111
19                                      Nagaland             69
20                                        Odisha            413
21                                   Pondicherry             29
22                                        Punjab            120
23                                     Rajasthan            140
24                                    Tamil Nadu              3
25                                     Tamilnadu            197
26                                     Telangana            469
27  The Dadra And Nagar Haveli And Daman And Diu             18
28                                       Tripura             36
29                                 Uttar Pradesh           1190
30                                   Uttarakhand            125
31                                   West Bengal            386
'''
'''
                                           state  missing_count
29                                 Uttar Pradesh           1190
26                                     Telangana            469
14                                     Karnataka            457
4                                          Bihar            421
20                                        Odisha            413
31                                   West Bengal            386
13                                     Jharkhand            310
3                                          Assam            257
16                                Madhya Pradesh            222
25                                     Tamilnadu            197
17                                   Maharashtra            188
12                             Jammu And Kashmir            161
6                                   Chhattisgarh            144
23                                     Rajasthan            140
30                                   Uttarakhand            125
22                                        Punjab            120
10                                       Haryana            114
18                                     Meghalaya            111
15                                        Kerala            107
9                                        Gujarat             91
7                                          Delhi             87
2                              Arunachal Pradesh             80
19                                      Nagaland             69
1                                 Andhra Pradesh             47
11                              Himachal Pradesh             46
28                                       Tripura             36
0                    Andaman And Nicobar Islands             33
21                                   Pondicherry             29
8                                            Goa             22
27  The Dadra And Nagar Haveli And Daman And Diu             18
5                                     Chandigarh              7
24                                    Tamil Nadu              3
'''

## ---------------------------------------------------------------------------------
# Merging data of residualsodiumcarbonate and gdp
## ---------------------------------------------------------------------------------

## Correction/Updation : Only taking data points after 1999 and using revised GDP after normalisation after taking ratio
import pandas as pd

# Assuming you have loaded your CSV data into pandas DataFrames
# df1 for residualsodiumcarbonate data
# df2 for GDP data

# Example data loading (replace with your actual data loading code)
df1 = pd.read_csv('has_data.csv')
df2 = pd.read_csv('NewGDPDataAfterNormalizationFrom1999.csv')

print(df1.columns)
print(df2.columns)
# Extract relevant columns from df2 for merging with df1
# Ensure state names from df1 match the column names in df2
states_to_merge = [state for state in df1['state'].unique() if state in df2.columns]
columns_to_merge = ['year'] + states_to_merge
df2_relevant = df2[columns_to_merge]

# Merge df1 and df2_relevant on 'year'
merged_df = pd.merge(df1, df2_relevant, on='year', how='left')

# Filter out rows where the GDP data for the corresponding state is missing
for state in states_to_merge:
    merged_df = merged_df[~((merged_df['state'] == state) & pd.isna(merged_df[state]))]

# Rename the GDP columns for clarity
merged_df.rename(columns={state: f'gdp_{state}' for state in states_to_merge}, inplace=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('merged_data.csv', index=False)

## ---------------------------------------------------------------------------------
# Removing uncessary columns
## ---------------------------------------------------------------------------------
import pandas as pd

# Load the CSV data
df = pd.read_csv('merged_data.csv')

# List of valid states based on GDP columns present in the DataFrame
valid_states = [col.split('_')[1] for col in df.columns if col.startswith('gdp_')]

# Filter the DataFrame to include only rows for valid states
df = df[df['state'].isin(valid_states)]

# Define a function to extract GDP for each row
def get_gdp(row):
    state = row['state']
    year = row['year']
    gdp_col = f'gdp_{state}'
    return row[gdp_col]

# Apply the function to create a new 'gdp' column
df['gdp'] = df.apply(get_gdp, axis=1)

# Create a new DataFrame with only the necessary columns
new_df = df[['state', 'district', 'year', 'residualsodiumcarbonate', 'gdp']]


# Filter rows where GDP is NaN or 0 (assuming valid GDP values are positive)
data_filtered = new_df[(new_df['gdp'].notna()) & (new_df['gdp'] > 0)]

# Print or save the filtered dataset
print(data_filtered)

# To save the filtered dataset back to a CSV file
data_filtered.to_csv('output_data.csv', index=False)


## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
# merging with gini Index (2011)
## ---------------------------------------------------------------------------------
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


## ---------------------------------------------------------------------------------
# Running regression
## ---------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load the CSV data into a DataFrame
df = pd.read_csv('output_data.csv')

# Display the first few rows of the cleaned DataFrame (optional)
print(df.head())

# Define the independent variable (gdp) and dependent variable (residualsodiumcarbonate)
X = df['gdp']  # Independent variable (gdp)
y = df['residualsodiumcarbonate']  # Dependent variable (residualsodiumcarbonate)

# Add constant to the independent variable (for intercept)
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the regression results
print(model.summary())

## ---------------------------------------------------------------------------------
# Removing data with empty gini value
## ---------------------------------------------------------------------------------
import pandas as pd

# Load the CSV file into a DataFrame
df1 = pd.read_csv('output_data_with_gini.csv')

# Filter out rows where 'Gini' is missing or NaN
df = df1.dropna(subset=['Gini'])

# Optionally, you can also filter out rows where 'Gini' is empty or zero (if applicable)
# df_cleaned = df[df['Gini'].notna() & (df['Gini'] != 0)]

# Save the cleaned DataFrame back to a new CSV file
df.to_csv('cleaned_data_without_nonempty_gini.csv', index=False)
print("Successfully removed entries with empty gini value, new saved saved in cleaned_data_without_nonempty_gini.csv")

## ---------------------------------------------------------------------------------
# Plotting regression
## ---------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# reading data from the csv
data = pd.read_csv("output_data.csv")

# Clean data and convert to numeric
data['gdp'] = pd.to_numeric(data['gdp'], errors='coerce')
data['residualsodiumcarbonate'] = pd.to_numeric(data['residualsodiumcarbonate'], errors='coerce')

# Drop rows with NaN values
data.dropna(subset=['gdp', 'residualsodiumcarbonate'], inplace=True)

# plotting the original values
x = data['gdp']
y = data['residualsodiumcarbonate']
plt.scatter(x, y)

# finding the maximum and minimum
# values of x, to get the
# range of data
max_x = x.max()
min_x = x.min()

# range of values for plotting
# the regression line
x_values = np.arange(min_x, max_x, 1)

# the substituted equation
y_values =  -2.187e-05 * x_values + 9.7100

# plotting the regression line
plt.plot(x_values, y_values, 'r')
plt.show()

## ---------------------------------------------------------------------------------
# Q2
## ---------------------------------------------------------------------------------

import matplotlib.pyplot as plt

# Define the independent variable (SDP) and dependent variable (residuals)
X = df['gdp']
y = df['residualsodiumcarbonate']

# Add constant to the independent variable (for intercept)
X = sm.add_constant(X)

# Predict residuals using the linear regression model
predictions = model.predict(X)

# Calculate residuals
residuals = y - predictions

# Store residuals in an array
residuals_array = residuals.values

# Plot 1: Model Residuals vs. Groundwater Quality Indicator (SDP)
plt.figure(figsize=(10, 6))
plt.scatter(X['gdp'], y, color='blue', label='Actual Data')
plt.scatter(X['gdp'], predictions, color='red', label='Predicted Data')
plt.xlabel('SDP (GDP)')
plt.ylabel('Groundwater Quality Indicator (Residual Sodium Carbonate)')
plt.title('Actual vs. Predicted Groundwater Quality Indicator')
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Model Residuals vs. SDP
plt.figure(figsize=(10, 6))
plt.scatter(X['gdp'], residuals, color='green')
plt.xlabel('SDP (GDP)')
plt.ylabel('Residuals (Model Errors)')
plt.title('Model Residuals vs. SDP')
plt.grid(True)
plt.show()

# Print and store residuals in an array
print("Residuals:", residuals_array)


# Plot histogram of model residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals_array, bins=30, color='red', edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Model Residuals')
plt.grid(True)
plt.show()

# Verify that the sum of residuals equals zero
print("Sum of residuals:", sum(residuals_array))







