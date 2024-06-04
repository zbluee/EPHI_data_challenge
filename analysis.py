import pandas as pd 

#1 Preprocessing and EDA:
#1A display the first few rows of the CHAMPS dataset
data_frame = pd.read_csv('src/CHAMPS.csv')
print(data_frame.head(), '\n')

#1B Get the number of rows and columns
num_rows, num_cols = data_frame.shape
print(f'The CHAMPS dataset dataset has {num_rows} rows and {num_cols} columns.', '\n')

#1C Enumerate the columns of the dataset
print(data_frame.columns.tolist(), '\n')

#D Rename the columns based on the provided mappings
data_frame.rename(columns={'champs_id':'CHAMPS_ID (Mortality)', 'dp_013' : 'case_type', 'dp_018' : 'underlying_cause', 'dp_118' : 'Main_maternal_condition'}, inplace=True)

#E Rename the values in the case_type column.
case_type_mapping = {
    'CH00716': 'Stillbirth',
    'CH01404': 'Death in the first 24 hours',
    'CH01405': 'Early Neonate (1 to 6 days)',
    'CH01406': 'Late Neonate (7 to 27 days)',
    'CH00718': 'Infant (28 days to less than 12 months)',
    'CH00719': 'Child (12 months to less than 60 months)'
}

data_frame['case_type'] = data_frame['case_type'].map(case_type_mapping)

#F Show the proportion of null values in each column.
null_proportion = data_frame.isnull().mean() * 100
print(null_proportion, '\n')

#Descriptive Data analysis
#2A What are the magnitude and proportion of each of the infant underlying cause for child death?
infant_underlying_cause_counts = data_frame['underlying_cause'].value_counts()
infant_underlying_cause_proportion = data_frame['underlying_cause'].value_counts(normalize=True) * 100

print('magnitude of each of the infant underlying cause for child death', infant_underlying_cause_counts, '\n')
print('Proportion of each of the infant underlying cause for child death', infant_underlying_cause_proportion, '\n')

#2B What are the proportion and magnitude of the maternal factors contributing for child death?
maternal_condition_counts = data_frame['Main_maternal_condition'].value_counts()
maternal_condition_proportions = data_frame['Main_maternal_condition'].value_counts(normalize =True) * 100

print('magnitude of the maternal factors contributing for child death', maternal_condition_counts, '\n')
print('proportion of the maternal factors contributing for child death', maternal_condition_proportions, '\n')

#2C What are the proportion of the child death by the case type
case_type_counts = data_frame['case_type'].value_counts()
case_type_proportion = data_frame['case_type'].value_counts(normalize= True) * 100

print('magnitude of the child death by the case type', case_type_counts, '\n')
print('proportion of the child death by the case type', case_type_proportion, '\n')

#3 Correlation analysis:
#Correlation Heatmap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
data_frame['case_type_encoded'] = label_encoder.fit_transform(data_frame['case_type'].astype(str))
data_frame['underlying_cause_encoded'] = label_encoder.fit_transform(data_frame['underlying_cause'].astype(str))
data_frame['maternal_condition_encoded'] = label_encoder.fit_transform(data_frame['Main_maternal_condition'].astype(str))

# Select only numerical columns for correlation analysis
numerical_data_frame = data_frame[['case_type_encoded', 'maternal_condition_encoded']]
#create correlation matrix
correlation_matrix = numerical_data_frame.corr()

#plot the Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot= True, cmap = 'coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


