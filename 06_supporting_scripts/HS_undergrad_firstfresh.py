import pandas as pd

# HS Data
high_school_data = pd.read_excel("pivoted_rc-ibhe-trend-data.xlsx")
high_school_data['Combined_Dropout_Rate'] = high_school_data['Dropout Rate']

# Convert percentage strings to numerical values
high_school_data['Combined_Dropout_Rate'] = high_school_data['Combined_Dropout_Rate'].str.rstrip('%').astype(float) / 100
high_school_data['Graduates'] = high_school_data["Enrollment:  Total Number"].str.replace(',','').astype(int) * (1 - high_school_data["Combined_Dropout_Rate"])
high_school_data["Graduates: White"] = high_school_data["Enrollment:  White"].str.rstrip("%").astype(float) * high_school_data['Graduates']
highsch_df = high_school_data[["School Year", "Graduates: White","Enrollment:  Total Number", "Combined_Dropout_Rate", "Graduates"]]
highsch_df = highsch_df.rename(columns={"School Year": "YEAR"})
highsch_df["YEAR"] = highsch_df["YEAR"].astype(int)

# Read Enrollments
df = pd.read_csv("FirstTimeFreshmenSince2009_202403241441.csv")
for index, row in df.iterrows():
    term_cd_str = str(int(row['TERM_CD']))  # Convert term code to string
    if term_cd_str[-1] == '8':  # Check the last character of the string
        df.at[index, 'YEAR'] = int(term_cd_str[1:-1])
    else:
        df.at[index, 'YEAR'] = int(term_cd_str[1:-1]) - 1

# Convert 'YEAR' column to integer
df['YEAR'] = df['YEAR'].astype(int)

yearly_enrollments = df.groupby(['YEAR', 'ADM_APPL_COLL_1_NAME'])['EDW_PERS_ID'].nunique().reset_index()

filtered = yearly_enrollments[yearly_enrollments["YEAR"]<2023]

filtered_hs = pd.merge(filtered, highsch_df, on="YEAR", how="left")

print(filtered_hs)

def calculate_correlation(group):
    return group['EDW_PERS_ID'].corr(group['Graduates'])

group_correlation = filtered_hs.groupby('ADM_APPL_COLL_1_NAME', as_index=False).apply(calculate_correlation)

print(group_correlation)