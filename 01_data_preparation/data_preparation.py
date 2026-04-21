"""
DATA PREPARATION FOR UIC STUDENT ENROLLMENT PREDICTION
=======================================================
Author: Tayler Erbe
Date: 2023

Description:
    This script reads raw UIC student enrollment data (sourced from a KNIME
    data wrangling workflow) and performs the following preparation steps:

        1. Feature engineering on term/year codes
        2. Separation of students by level (graduate, undergrad, professional, etc.)
        3. Creation of enrollment history features (terms completed, years completed)
        4. Labeling of target variables for:
               - Next term enrollment (0=dropped, 1=continued, 2=graduated)
               - Next year enrollment (any term)
               - Next fall year enrollment
               - Binary success outcomes (0=unsuccessful, 1=successful/graduated)
        5. Aggregation to one record per student per academic year
        6. Export of all processed datasets as Parquet files

Input:
    ALL_DATA_6162023.parquet  (produced by the KNIME data wrangling workflow)

Outputs:
    GRADUATE_BY_TERM.parquet
    UNDERGRADUATE_BY_TERM.parquet
    PROFESSIONAL_BY_TERM.parquet
    LAW_BY_TERM.parquet
    NON_CRED_BY_TERM.parquet
    SCALES_BY_TERM.parquet
    COMBINED_BY_TERM.parquet
    GRADUATE_TARGET_BY_TERM.parquet          ... (and equivalents per level)
    GRADUATE_TARGET_BY_YEAR.parquet          ... (and equivalents per level)
    COMBINED_TARGET_BY_YEAR.parquet
    GRADUATE_TARGET_BY_YEAR_SINGLE_RANDOM_POINT.parquet  ... (and equivalents)
    COMBINED_TARGET_BY_YEAR_SINGLE_RANDOM_POINT.parquet
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =============================================================================
# 1. LOAD DATA
# =============================================================================
# Raw data was created inside of KNIME — see the KNIME workflow in this folder.
# The file contains one row per student per enrollment snapshot (term).

df = pd.read_parquet('ALL_DATA_6162023.parquet')

print(f"Loaded {len(df):,} rows x {df.shape[1]} columns")
print("\nStudent level distribution:")
print(df["STUDENT_LEVEL_DESC"].value_counts())
print("\nAcademic year distribution:")
print(df["ACAD_YR_CD"].value_counts())


# =============================================================================
# 2. FEATURE ENGINEERING — YEARS AWAY FROM GRADUATION
# =============================================================================
# Calculates how many academic years remain until a student's expected
# graduation date relative to their current academic year.

df['ACAD_YR_CD'] = df['ACAD_YR_CD'].fillna(0)
df['EXPCT_GRAD_TERM_CD'] = df['EXPCT_GRAD_TERM_CD'].fillna(0)
df['TERM_CD'] = df['TERM_CD'].fillna(0)
df['EXPCT_GRAD_ACAD_YR_CD'] = df['EXPCT_GRAD_ACAD_YR_CD'].fillna(0)

# Convert to numeric, coercing any bad values to NaN
for col in ['ACAD_YR_CD', 'EXPCT_GRAD_TERM_CD', 'TERM_CD', 'EXPCT_GRAD_ACAD_YR_CD']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Round and cast term/year columns to integer strings
df['ACAD_YR_CD'] = df['ACAD_YR_CD'].round().astype(int).astype(str)
df['EXPCT_GRAD_TERM_CD'] = df['EXPCT_GRAD_TERM_CD'].round().astype(int).astype(str)
df['TERM_CD'] = df['TERM_CD'].round().astype(int).astype(str)

# Handle infinity and NaN in expected graduation year
df['EXPCT_GRAD_ACAD_YR_CD'] = df['EXPCT_GRAD_ACAD_YR_CD'].replace([np.inf, -np.inf], np.nan)
df['EXPCT_GRAD_ACAD_YR_CD'] = pd.to_numeric(df['EXPCT_GRAD_ACAD_YR_CD'], errors='coerce')
df['EXPCT_GRAD_ACAD_YR_CD'] = df['EXPCT_GRAD_ACAD_YR_CD'].round().astype('Int64')
df['EXPCT_GRAD_ACAD_YR_CD'] = pd.to_numeric(df['EXPCT_GRAD_ACAD_YR_CD'], errors='coerce').fillna(0)

df['ACAD_YR_CD'] = pd.to_numeric(df['ACAD_YR_CD'], errors='coerce').fillna(0)

# Years away = difference in the first two digits of the academic year code
# e.g. EXPCT_GRAD_ACAD_YR_CD=2425 -> 24, ACAD_YR_CD=2223 -> 22 => 2 years away
df["YEARS_AWAY_FROM_GRAD"] = (
    (df['EXPCT_GRAD_ACAD_YR_CD'] // 100) - (df['ACAD_YR_CD'] // 100)
)
df.loc[df["YEARS_AWAY_FROM_GRAD"] < 0, "YEARS_AWAY_FROM_GRAD"] = 0


# =============================================================================
# 3. FEATURE ENGINEERING — YEAR AND SEMESTER CODES FROM TERM_CD
# =============================================================================
# TERM_CD format: [college_prefix][4-digit year][semester_digit]
# Semester digit: 1 = spring, 5 = summer, 8 = fall

df['TERM_CD'] = df['TERM_CD'].astype(str)
df['YEAR_CD'] = df['TERM_CD'].str.slice(1, 5)
df['SEMESTER_CD'] = df['TERM_CD'].str.extract(r'(\d)$')


# =============================================================================
# 4. FEATURE ENGINEERING — ACADEMIC YEAR CODE LABEL (ACAD_YEAR_CODE)
# =============================================================================
# Maps numeric TERM_CD values to a human-readable academic year label.
# e.g. TERM_CD 220221 -> ACAD_YEAR_CODE "2122" (AY 2021-2022)

def map_term_to_acad_year(df):
    df['TERM_CD'] = df['TERM_CD'].astype(int)
    df['TERM_CD'] = df['TERM_CD'].round().astype(str)

    replace_dict = {
        "220131": "1213", "220128": "1213", "220135": "1213",
        "220141": "1314", "220138": "1314", "220145": "1314",
        "220151": "1415", "220148": "1415", "220155": "1415",
        "220161": "1516", "220158": "1516", "220165": "1516",
        "220171": "1617", "220168": "1617", "220175": "1617",
        "220181": "1718", "220178": "1718", "220185": "1718",
        "220191": "1819", "220188": "1819", "220195": "1819",
        "220201": "1920", "220198": "1920", "220205": "1920",
        "220211": "2021", "220208": "2021", "220215": "2021",
        "220221": "2122", "220218": "2122", "220225": "2122",
        "220231": "2223", "220228": "2223", "220235": "2223",
        "220238": "2324", "220241": "2324", "220245": "2324",
    }

    df["ACAD_YEAR_CODE"] = df["TERM_CD"].replace(replace_dict)
    return df

df = map_term_to_acad_year(df)


# =============================================================================
# 5. SEPARATE DATA BY STUDENT LEVEL
# =============================================================================
# Split the main dataframe into one dataframe per student classification level.

graduate_df = df[df["STUDENT_LEVEL_DESC"].isin([
    'Graduate - Chicago', 'Graduate Non-Degree Chicago', 'Graduate Online – Chicago'
])].copy()

undergrad_df = df[df["STUDENT_LEVEL_DESC"].isin([
    'Undergrad - Chicago', 'Undergrad Non-Degree Chicago'
])].copy()

professional_df = df[df["STUDENT_LEVEL_DESC"].isin(['Professional - Chicago'])].copy()
law_df          = df[df["STUDENT_LEVEL_DESC"].isin(['Law - Chicago'])].copy()
non_cred_df     = df[df["STUDENT_LEVEL_DESC"].isin(['Non-Credit - Chicago'])].copy()
scales_df       = df[df["STUDENT_LEVEL_DESC"].isin(['SCALES - Chicago'])].copy()

all_level_dfs = [graduate_df, undergrad_df, professional_df, law_df, non_cred_df, scales_df]


# =============================================================================
# 6. FEATURE ENGINEERING — ENROLLMENT HISTORY LISTS
# =============================================================================

def concatenate_unique_terms(df):
    """
    For each student (EDW_PERS_ID), concatenates all unique TERM_CDs they were
    ever enrolled in into a single comma-separated string.

    Example: student enrolled in terms 220218, 220221, 220228 ->
             UNIQUE_TERM_CD = "220218,220221,220228"
    """
    unique_terms = (
        df.groupby('EDW_PERS_ID')['TERM_CD']
        .unique()
        .apply(lambda x: ','.join(x))
        .reset_index()
        .rename(columns={'TERM_CD': 'UNIQUE_TERM_CD'})
    )
    return df.merge(unique_terms, on='EDW_PERS_ID')


def concatenate_unique_years(df):
    """
    For each student, concatenates all unique academic years (ACAD_YEAR_CODE)
    they were ever enrolled in into a single comma-separated string.

    Example: student in AY 2122 and 2223 ->
             AH_ACAD_YEAR_CD_LIST = "2122,2223"
    """
    unique_years = (
        df.groupby('EDW_PERS_ID')['ACAD_YEAR_CODE']
        .unique()
        .apply(lambda x: ','.join(x))
        .reset_index()
        .rename(columns={'ACAD_YEAR_CODE': 'AH_ACAD_YEAR_CD_LIST'})
    )
    return df.merge(unique_years, on='EDW_PERS_ID')


for i, level_df in enumerate(all_level_dfs):
    all_level_dfs[i] = concatenate_unique_terms(level_df)
    all_level_dfs[i] = concatenate_unique_years(all_level_dfs[i])

graduate_df, undergrad_df, professional_df, law_df, non_cred_df, scales_df = all_level_dfs


# =============================================================================
# 7. FEATURE ENGINEERING — TOTAL AND CURRENT COMPLETED TERMS
# =============================================================================

def calculate_number_terms(series):
    """
    Counts the number of terms in each student's UNIQUE_TERM_CD string.
    Returns TOTAL_COMPLETED_TERMS — the total number of terms ever enrolled.
    """
    return [len(s.split(",")) for s in series]


def calculate_current_completed_terms(df):
    """
    For each row, calculates how many terms the student had completed up to
    and including the TERM_CD on that row (i.e. the sequential count of terms).

    Example: if a student enrolled in terms [220218, 220221, 220228]:
        - Row with TERM_CD=220218 -> CURRENT_COMPLETED_TERMS = 1
        - Row with TERM_CD=220221 -> CURRENT_COMPLETED_TERMS = 2
        - Row with TERM_CD=220228 -> CURRENT_COMPLETED_TERMS = 3
    """
    df["CURRENT_COMPLETED_TERMS"] = 1
    unique_ids = df["EDW_PERS_ID"].unique()

    for uid in unique_ids:
        temp_df = df[df["EDW_PERS_ID"] == uid]
        temp_idx = temp_df.index.values
        sorted_terms = sorted(set(temp_df["TERM_CD"]))
        term_rank = {term: rank + 1 for rank, term in enumerate(sorted_terms)}

        for idx in temp_idx:
            term = df.at[idx, "TERM_CD"]
            if term in term_rank:
                df.at[idx, "CURRENT_COMPLETED_TERMS"] = term_rank[term]

    return df


def calculate_completed_academic_years(df):
    """
    For each row, calculates how many academic years the student had completed
    up to and including the ACAD_YEAR_CODE on that row.

    Example: student in years [2122, 2223]:
        - Rows in AY 2122 -> CURRENT_COMPLETED_ACAD_YEARS = 1
        - Rows in AY 2223 -> CURRENT_COMPLETED_ACAD_YEARS = 2
    """
    df["ACAD_YEAR_CODE"] = df["ACAD_YEAR_CODE"].astype(int)
    df["CURRENT_COMPLETED_ACAD_YEARS"] = 1
    unique_ids = df["EDW_PERS_ID"].unique()

    for uid in unique_ids:
        temp_df = df[df["EDW_PERS_ID"] == uid]
        temp_idx = temp_df.index.values
        sorted_years = sorted(set(temp_df["ACAD_YEAR_CODE"]))
        year_rank = {year: rank + 1 for rank, year in enumerate(sorted_years)}

        for idx in temp_idx:
            year = df.at[idx, "ACAD_YEAR_CODE"]
            if year in year_rank:
                df.at[idx, "CURRENT_COMPLETED_ACAD_YEARS"] = year_rank[year]

    return df


for i, level_df in enumerate(all_level_dfs):
    all_level_dfs[i]["TOTAL_COMPLETED_TERMS"] = calculate_number_terms(level_df["UNIQUE_TERM_CD"])
    all_level_dfs[i] = calculate_current_completed_terms(all_level_dfs[i])
    all_level_dfs[i] = calculate_completed_academic_years(all_level_dfs[i])

graduate_df, undergrad_df, professional_df, law_df, non_cred_df, scales_df = all_level_dfs


# =============================================================================
# 8. EXPORT — BY TERM (before target labeling)
# =============================================================================

graduate_df.to_parquet("GRADUATE_BY_TERM.parquet",      engine='pyarrow', compression='snappy')
undergrad_df.to_parquet("UNDERGRADUATE_BY_TERM.parquet", engine='pyarrow', compression='snappy')
professional_df.to_parquet("PROFESSIONAL_BY_TERM.parquet", engine='pyarrow', compression='snappy')
law_df.to_parquet("LAW_BY_TERM.parquet",                engine='pyarrow', compression='snappy')
non_cred_df.to_parquet("NON_CRED_BY_TERM.parquet",      engine='pyarrow', compression='snappy')
scales_df.to_parquet("SCALES_BY_TERM.parquet",          engine='pyarrow', compression='snappy')

pd.concat(all_level_dfs, ignore_index=True).to_parquet(
    "COMBINED_BY_TERM.parquet", engine='pyarrow', compression='snappy'
)
print("Exported BY_TERM files.")


# =============================================================================
# 9. TARGET VARIABLE LABELING — NEXT TERM ENROLLMENT
# =============================================================================
# Labels each row based on whether the student enrolled in the next term.
#
# Label logic:
#   1 = Student enrolled in the immediately following term
#       (fall -> spring, spring -> fall, summer -> fall)
#   2 = Student did NOT enroll next term, but has a degree (graduated)
#   0 = Student did not enroll next term and did not graduate (discontinued)
#
# Note: Rows in AY 2223 (the most recent year) are labeled 1 by default
#       because we cannot observe future enrollment from the data cutoff.

def label_next_term_target_variables(df):
    def get_next_term(current_term):
        current_term = str(current_term)
        college  = current_term[0:1]
        year     = current_term[1:5]
        semester = current_term[5:6]

        if semester == "8":    # Fall -> next Spring
            return [college + str(int(year) + 1) + "1"]
        elif semester == "1":  # Spring -> next Fall (same calendar year)
            return [college + str(int(year)) + "8"]
        elif semester == "5":  # Summer -> next Fall (same calendar year)
            return [college + str(int(year)) + "8"]

    df.reset_index(drop=True, inplace=True)

    target = []
    for i in range(len(df)):
        next_terms = get_next_term(df["TERM_CD"][i])
        enrolled = any(t in df["UNIQUE_TERM_CD"][i].split(',') for t in next_terms)
        target.append('1' if enrolled else '0')

    df["TARGET_ENROLLED_NEXT_TERM"] = target

    for i in range(len(df)):
        # Override: student graduated (has a current degree indicator) but didn't re-enroll
        if df['STUDENT_AH_DEG_CUR_INFO_IND'][i] == 'Y' and df["TARGET_ENROLLED_NEXT_TERM"][i] == '0':
            df.at[i, "TARGET_ENROLLED_NEXT_TERM"] = '2'
        # Override: most recent year — assume continued (future unknown)
        if df['ACAD_YEAR_CODE'][i] == 2223:
            df.at[i, "TARGET_ENROLLED_NEXT_TERM"] = '1'

    return df


# =============================================================================
# 10. TARGET VARIABLE LABELING — NEXT YEAR ENROLLMENT (ANY TERM)
# =============================================================================
# Labels each row based on whether the student enrolled in ANY term
# during the next academic year. This is more lenient than next-term labeling —
# a gap semester is tolerated as long as they returned within the year.

def label_next_year_target_variables(df):
    def get_next_year_terms(current_term):
        current_term = str(current_term)
        college  = current_term[0:1]
        year     = current_term[1:5]
        semester = current_term[5:6]

        if semester == "8":  # Fall -> look at fall/spring/summer of next year
            return [
                college + str(int(year) + 1) + "8",
                college + str(int(year) + 2) + "1",
                college + str(int(year) + 2) + "5",
            ]
        else:  # Spring or Summer -> look at fall/spring/summer of same next year
            return [
                college + str(int(year)) + "8",
                college + str(int(year) + 1) + "1",
                college + str(int(year) + 1) + "5",
            ]

    df.reset_index(drop=True, inplace=True)

    target = []
    for i in range(len(df)):
        next_terms = get_next_year_terms(df["TERM_CD"][i])
        enrolled = any(t in df["UNIQUE_TERM_CD"][i].split(',') for t in next_terms)
        target.append('1' if enrolled else '0')

    df["TARGET_ENROLLED_NEXT_YEAR"] = target

    for i in range(len(df)):
        if df['STUDENT_AH_DEG_CUR_INFO_IND'][i] == 'Y' and df["TARGET_ENROLLED_NEXT_YEAR"][i] == '0':
            df.at[i, "TARGET_ENROLLED_NEXT_YEAR"] = '2'
        if df['ACAD_YEAR_CODE'][i] == 2223:
            df.at[i, "TARGET_ENROLLED_NEXT_YEAR"] = '1'

    return df


# =============================================================================
# 11. TARGET VARIABLE LABELING — NEXT FALL YEAR ENROLLMENT
# =============================================================================
# Labels each row based on whether the student enrolled in the next fall term
# specifically. Used for fall-to-fall retention modeling.

def label_next_fall_year_target_variables(df):
    def get_next_fall_term(current_term):
        current_term = str(current_term)
        college  = current_term[0:1]
        year     = current_term[1:5]
        semester = current_term[5:6]

        if semester == "8":  # Fall -> next fall
            return [college + str(int(year) + 1) + "8"]
        else:                # Spring or Summer -> same year's fall
            return [college + str(int(year)) + "8"]

    df.reset_index(drop=True, inplace=True)

    target = []
    for i in range(len(df)):
        next_terms = get_next_fall_term(df["TERM_CD"][i])
        enrolled = any(t in df["UNIQUE_TERM_CD"][i].split(',') for t in next_terms)
        target.append('1' if enrolled else '0')

    df["TARGET_ENROLLED_NEXT_FALL_YEAR"] = target

    for i in range(len(df)):
        if df['STUDENT_AH_DEG_CUR_INFO_IND'][i] == 'Y' and df["TARGET_ENROLLED_NEXT_FALL_YEAR"][i] == '0':
            df.at[i, "TARGET_ENROLLED_NEXT_FALL_YEAR"] = '2'
        if df['ACAD_YEAR_CODE'][i] == 2223:
            df.at[i, "TARGET_ENROLLED_NEXT_FALL_YEAR"] = '1'

    return df


# =============================================================================
# 12. TARGET VARIABLE LABELING — BINARY SUCCESS OUTCOMES
# =============================================================================
# Collapses the 3-class target (0=discontinued, 1=continued, 2=graduated) into
# a binary outcome: 0 = unsuccessful (discontinued), 1 = successful (continued OR graduated).

def successful_student_target_by_term(df):
    df["SUCCESSFUL_OUTCOME_TARGET_BY_TERM"] = df["TARGET_ENROLLED_NEXT_TERM"].copy()
    df.loc[df["SUCCESSFUL_OUTCOME_TARGET_BY_TERM"] == "2", "SUCCESSFUL_OUTCOME_TARGET_BY_TERM"] = "1"
    return df


def successful_student_target_by_year(df):
    df["SUCCESSFUL_OUTCOME_TARGET_BY_YEAR"] = df["TARGET_ENROLLED_NEXT_YEAR"].copy()
    df.loc[df["SUCCESSFUL_OUTCOME_TARGET_BY_YEAR"] == "2", "SUCCESSFUL_OUTCOME_TARGET_BY_YEAR"] = "1"
    return df


# Apply all target labeling functions to each student level dataframe
for i, level_df in enumerate(all_level_dfs):
    all_level_dfs[i] = label_next_term_target_variables(level_df)
    all_level_dfs[i] = successful_student_target_by_term(all_level_dfs[i])
    all_level_dfs[i] = label_next_year_target_variables(all_level_dfs[i])
    all_level_dfs[i] = successful_student_target_by_year(all_level_dfs[i])
    all_level_dfs[i] = label_next_fall_year_target_variables(all_level_dfs[i])

graduate_df, undergrad_df, professional_df, law_df, non_cred_df, scales_df = all_level_dfs


# =============================================================================
# 13. EXPORT — TARGET BY TERM (all term rows, with labels)
# =============================================================================

graduate_df.to_parquet("GRADUATE_TARGET_BY_TERM.parquet",       engine='pyarrow', compression='snappy')
undergrad_df.to_parquet("UNDERGRADUATE_TARGET_BY_TERM.parquet", engine='pyarrow', compression='snappy')
professional_df.to_parquet("PROFESSIONAL_TARGET_BY_TERM.parquet", engine='pyarrow', compression='snappy')
law_df.to_parquet("LAW_TARGET_BY_TERM.parquet",                 engine='pyarrow', compression='snappy')
non_cred_df.to_parquet("NON_CRED_TARGET_BY_TERM.parquet",       engine='pyarrow', compression='snappy')
scales_df.to_parquet("SCALES_TARGET_BY_TERM.parquet",           engine='pyarrow', compression='snappy')
print("Exported TARGET_BY_TERM files.")


# =============================================================================
# 14. AGGREGATE TO ONE ROW PER STUDENT PER ACADEMIC YEAR
# =============================================================================
# Reduces multiple term snapshots within the same academic year down to one row
# per student per year, keeping the snapshot from the most recent semester.
#
# Semester priority (kept = lowest mapped value):
#   Summer (5) -> mapped to 1 (most recent within year)
#   Spring (1) -> mapped to 2
#   Fall   (8) -> mapped to 3 (least recent — start of year)

def aggregate_to_one_row_per_year(df):
    df['SEMESTER_CD'] = df['SEMESTER_CD'].astype(int)
    semester_priority = {1: 2, 8: 3, 5: 1}
    df['SEMESTER_CD'] = df['SEMESTER_CD'].replace(semester_priority)

    # Keep the row with the lowest priority value (most recent semester) per student per year
    min_idx = df.groupby(['EDW_PERS_ID', 'ACAD_YEAR_CODE'])['SEMESTER_CD'].idxmin()
    return df.loc[min_idx]


for i, level_df in enumerate(all_level_dfs):
    all_level_dfs[i] = aggregate_to_one_row_per_year(level_df)

graduate_df, undergrad_df, professional_df, law_df, non_cred_df, scales_df = all_level_dfs


# =============================================================================
# 15. EXPORT — TARGET BY YEAR (one row per student per academic year)
# =============================================================================

graduate_df.to_parquet("GRADUATE_TARGET_BY_YEAR.parquet",       engine='pyarrow', compression='snappy')
undergrad_df.to_parquet("UNDERGRADUATE_TARGET_BY_YEAR.parquet", engine='pyarrow', compression='snappy')
professional_df.to_parquet("PROFESSIONAL_TARGET_BY_YEAR.parquet", engine='pyarrow', compression='snappy')
law_df.to_parquet("LAW_TARGET_BY_YEAR.parquet",                 engine='pyarrow', compression='snappy')
non_cred_df.to_parquet("NON_CRED_TARGET_BY_YEAR.parquet",       engine='pyarrow', compression='snappy')
scales_df.to_parquet("SCALES_TARGET_BY_YEAR.parquet",           engine='pyarrow', compression='snappy')

pd.concat(all_level_dfs).to_parquet(
    "COMBINED_TARGET_BY_YEAR.parquet", engine='pyarrow', compression='snappy'
)
print("Exported TARGET_BY_YEAR files.")


# =============================================================================
# 16. SINGLE RANDOM POINT — ONE ROW PER STUDENT (for model training)
# =============================================================================
# Takes one random snapshot per student across all years.
# This prevents data leakage and avoids confusing the model with multiple
# rows for the same student that may have different target labels depending
# on the year observed.

def keep_single_point_data(df):
    """
    Randomly selects one row per unique EDW_PERS_ID.
    Uses random_state=42 for reproducibility.
    """
    shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return shuffled.groupby("EDW_PERS_ID").first().reset_index()


single_point_dfs = [keep_single_point_data(df) for df in all_level_dfs]
(
    new_graduate_df, new_undergrad_df, new_professional_df,
    new_law_df, new_non_cred_df, new_scales_df
) = single_point_dfs


# =============================================================================
# 17. EXPORT — SINGLE RANDOM POINT PER STUDENT
# =============================================================================

new_graduate_df.to_parquet("GRADUATE_TARGET_BY_YEAR_SINGLE_RANDOM_POINT.parquet",       engine='pyarrow', compression='snappy')
new_undergrad_df.to_parquet("UNDERGRADUATE_TARGET_BY_YEAR_SINGLE_RANDOM_POINT.parquet", engine='pyarrow', compression='snappy')
new_professional_df.to_parquet("PROFESSIONAL_TARGET_BY_YEAR_SINGLE_RANDOM_POINT.parquet", engine='pyarrow', compression='snappy')
new_law_df.to_parquet("LAW_TARGET_BY_YEAR_SINGLE_RANDOM_POINT.parquet",                 engine='pyarrow', compression='snappy')
new_non_cred_df.to_parquet("NON_CRED_TARGET_BY_YEAR_SINGLE_RANDOM_POINT.parquet",       engine='pyarrow', compression='snappy')
new_scales_df.to_parquet("SCALES_TARGET_BY_YEAR_SINGLE_RANDOM_POINT.parquet",           engine='pyarrow', compression='snappy')

pd.concat(single_point_dfs, ignore_index=True).to_parquet(
    "COMBINED_TARGET_BY_YEAR_SINGLE_RANDOM_POINT.parquet", engine='pyarrow', compression='snappy'
)
print("Exported SINGLE_RANDOM_POINT files.")
print("\nData preparation complete.")
