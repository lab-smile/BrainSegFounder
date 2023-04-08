import pandas as pd

# Read in the CSV file
path = "/red/ruogu.fang/jbroce/ADNI_3D/labels.csv"

def subject_column(path):
    df = pd.read_csv(path)

# Create a new column called "subject"

    df[['subject', 'scan']] = df['id'].str.split('M', expand=True)
    return df[['id', 'subject', 'label']]
# Print out the resulting dataframe
df = subject_column(path)
print(df['id'].str.split('M', expand=True)[1].tolist())
