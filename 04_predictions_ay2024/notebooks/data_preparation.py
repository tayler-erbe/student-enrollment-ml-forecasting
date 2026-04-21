"""
data_preparation.py
====================
Student enrollment data preparation pipeline.

This script processes raw student registration snapshots and produces
labeled datasets for retention/graduation modeling. It was originally
developed inside a KNIME workflow; this file is the standalone Python
equivalent.

Pipeline overview
-----------------
1.  Load raw parquet snapshots and merge college-code data.
2.  Engineer term/year features and compute graduation-proximity fields.
3.  Split records by student level (graduate, undergrad, etc.).
4.  Build historical enrollment histories (unique terms & years per student).
5.  Calculate rolling enrollment counts (total and per-current-term/year).
6.  Label three sets of target variables:
      - TARGET_ENROLLED_NEXT_TERM   : enrolled in the immediate next term?
      - TARGET_ENROLLED_NEXT_YEAR   : enrolled in any term next year?
      - TARGET_ENROLLED_NEXT_FALL_YEAR : enrolled in next fall specifically?
    Labels: 0 = dropped out, 1 = continued, 2 = graduated
7.  Derive binary "successful outcome" flags (merge labels 1 & 2 → 1).
8.  Aggregate to one row per student per academic year (most-recent semester).
9.  Optionally downsample to a single random row per student.
10. Export all segment-level and combined parquet files.

Dependencies: pandas, numpy, scikit-learn (imported but unused here),
              pyarrow (for parquet I/O)
"""

# ---------------------------------------------------------------------------
# 0. IMPORTS
# ---------------------------------------------------------------------------
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier      # reserved for modelling step
from sklearn.metrics import accuracy_score               # reserved for modelling step
from sklearn.model_selection import train_test_split     # reserved for modelling step


# ---------------------------------------------------------------------------
# 1. COMPUTE CURRENT ACADEMIC YEAR CODE
#    Academic year codes follow the pattern XXYY where XX = start year
#    (last two digits) and YY = end year. E.g. 2023–24 → "2324".
#    The prefix "23" is hardcoded per institutional convention.
# ---------------------------------------------------------------------------
current_year = datetime.now().year
current_yr_str = str(current_year)
current_acad_yr_cd_str = "23" + current_yr_str[2:4]   # e.g. "2324"
current_acad_yr_cd_int = int(current_acad_yr_cd_str)   # e.g. 2324


# ---------------------------------------------------------------------------
# 2. LOAD & MERGE RAW DATA
#    Two parquet sources are joined on student ID + registration snapshot key.
#    A duplicate COLL_CD column from the left table is dropped and the one
#    from the right table is kept.
# ---------------------------------------------------------------------------
A_df = pd.read_parquet("ALL_DATA_6162023.parquet")
B_df = pd.read_parquet("ALL_DATA_coll_cd.parquet")

df = pd.merge(
    A_df, B_df,
    how="left",
    on=["EDW_PERS_ID", "REG_SNAPSHOT_KEY"],
)
df = df.drop(columns=["COLL_CD_x"])
df = df.rename(columns={"COLL_CD_y": "COLL_CD"})

df.to_parquet("ALL_DATA_6292024.parquet")


# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING – YEARS AWAY FROM GRADUATION
#    Computes how many academic years remain before a student's expected
#    graduation.  Negative values (data anomalies) are floored at 0.
# ---------------------------------------------------------------------------
# Fill missing values before numeric conversion
for col in ["ACAD_YR_CD", "EXPCT_GRAD_TERM_CD", "TERM_CD", "EXPCT_GRAD_ACAD_YR_CD"]:
    df[col] = df[col].fillna(0)

# Convert to numeric; non-parseable entries become NaN
df["ACAD_YR_CD"] = pd.to_numeric(df["ACAD_YR_CD"], errors="coerce")
df["EXPCT_GRAD_TERM_CD"] = pd.to_numeric(df["EXPCT_GRAD_TERM_CD"], errors="coerce")
df["TERM_CD"] = pd.to_numeric(df["TERM_CD"], errors="coerce")
df["EXPCT_GRAD_ACAD_YR_CD"] = pd.to_numeric(df["EXPCT_GRAD_ACAD_YR_CD"], errors="coerce")

# Round and cast to string for downstream string-based term operations
df["ACAD_YR_CD"] = df["ACAD_YR_CD"].round().astype(int).astype(str)
df["EXPCT_GRAD_TERM_CD"] = df["EXPCT_GRAD_TERM_CD"].round().astype(int).astype(str)
df["TERM_CD"] = df["TERM_CD"].round().astype(int).astype(str)

# Handle infinities before converting EXPCT_GRAD_ACAD_YR_CD to nullable int
df["EXPCT_GRAD_ACAD_YR_CD"] = df["EXPCT_GRAD_ACAD_YR_CD"].replace(
    [np.inf, -np.inf], np.nan
)
df["EXPCT_GRAD_ACAD_YR_CD"] = pd.to_numeric(
    df["EXPCT_GRAD_ACAD_YR_CD"], errors="coerce"
).round().astype("Int64")

# Re-cast both year columns to float for arithmetic
df["EXPCT_GRAD_ACAD_YR_CD"] = pd.to_numeric(
    df["EXPCT_GRAD_ACAD_YR_CD"], errors="coerce"
).fillna(0)
df["ACAD_YR_CD"] = pd.to_numeric(df["ACAD_YR_CD"], errors="coerce").fillna(0)

# The academic year code encodes the century in the first two digits;
# integer-dividing by 100 extracts just the two-digit year portion.
df["YEARS_AWAY_FROM_GRAD"] = (
    df["EXPCT_GRAD_ACAD_YR_CD"] // 100
) - (df["ACAD_YR_CD"] // 100)

# Floor at 0 – students who have already passed their expected graduation
# date should not contribute negative values.
df.loc[df["YEARS_AWAY_FROM_GRAD"] < 0, "YEARS_AWAY_FROM_GRAD"] = 0


# ---------------------------------------------------------------------------
# 4. EXTRACT YEAR & SEMESTER CODES FROM TERM_CD
#    TERM_CD format: <college_digit><4-digit-year><semester_digit>
#      semester digits: 1 = spring, 5 = summer, 8 = fall
# ---------------------------------------------------------------------------
df["TERM_CD"] = df["TERM_CD"].astype(str)
df["YEAR_CD"] = df["TERM_CD"].str.slice(1, 5)           # characters 1–4
df["SEMESTER_CD"] = df["TERM_CD"].str.extract(r"(\d)$") # last digit


# ---------------------------------------------------------------------------
# 5. MAP TERM_CD TO HUMAN-READABLE ACADEMIC YEAR CODE
#    Each term code is mapped to a two-year academic year string, e.g.
#    "1314" for academic year 2013–14.
# ---------------------------------------------------------------------------
def map_term_to_acad_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an ACAD_YEAR_CODE column derived from TERM_CD.

    The replace_dict maps each known term code (as a string) to the
    corresponding academic year code string.
    """
    df["TERM_CD"] = df["TERM_CD"].round().astype(str) if df["TERM_CD"].dtype != object else df["TERM_CD"]

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


df["TERM_CD"] = df["TERM_CD"].astype(int)
df = map_term_to_acad_year(df)


# ---------------------------------------------------------------------------
# 6. SPLIT BY STUDENT LEVEL DESCRIPTION
#    Creates six segment-level DataFrames for independent downstream
#    processing.
# ---------------------------------------------------------------------------
graduate_df = df[df["STUDENT_LEVEL_DESC"].isin(
    ["Graduate - Chicago", "Graduate Non-Degree Chicago", "Graduate Online – Chicago"]
)].copy()

undergrad_df = df[df["STUDENT_LEVEL_DESC"].isin(
    ["Undergrad - Chicago", "Undergrad Non-Degree Chicago"]
)].copy()

professional_df = df[df["STUDENT_LEVEL_DESC"].isin(
    ["Professional - Chicago"]
)].copy()

law_df = df[df["STUDENT_LEVEL_DESC"].isin(
    ["Law - Chicago"]
)].copy()

non_cred_df = df[df["STUDENT_LEVEL_DESC"].isin(
    ["Non-Credit - Chicago"]
)].copy()

scales_df = df[df["STUDENT_LEVEL_DESC"].isin(
    ["SCALES - Chicago"]
)].copy()

# Convenience list for applying transformations to all segments at once
ALL_SEGMENTS = [graduate_df, undergrad_df, professional_df,
                law_df, non_cred_df, scales_df]


# ---------------------------------------------------------------------------
# 7. BUILD HISTORICAL ENROLLMENT HISTORIES
#    7a. UNIQUE_TERM_CD  – comma-separated list of all terms ever attended
#    7b. AH_ACAD_YEAR_CD_LIST – comma-separated list of all academic years
# ---------------------------------------------------------------------------
def concatenate_unique_terms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add UNIQUE_TERM_CD column: a comma-separated string of every unique
    TERM_CD value for each student (across all rows/years).
    """
    unique_terms = (
        df.groupby("EDW_PERS_ID")["TERM_CD"]
        .unique()
        .apply(lambda x: ",".join(x))
        .reset_index()
        .rename(columns={"TERM_CD": "UNIQUE_TERM_CD"})
    )
    return df.merge(unique_terms, on="EDW_PERS_ID")


def concatenate_unique_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add AH_ACAD_YEAR_CD_LIST column: a comma-separated string of every
    unique ACAD_YEAR_CODE value for each student (historical record).
    """
    unique_years = (
        df.groupby("EDW_PERS_ID")["ACAD_YEAR_CODE"]
        .unique()
        .apply(lambda x: ",".join(x))
        .reset_index()
        .rename(columns={"ACAD_YEAR_CODE": "AH_ACAD_YEAR_CD_LIST"})
    )
    return df.merge(unique_years, on="EDW_PERS_ID")


graduate_df     = concatenate_unique_terms(graduate_df)
undergrad_df    = concatenate_unique_terms(undergrad_df)
professional_df = concatenate_unique_terms(professional_df)
law_df          = concatenate_unique_terms(law_df)
non_cred_df     = concatenate_unique_terms(non_cred_df)
scales_df       = concatenate_unique_terms(scales_df)

graduate_df     = concatenate_unique_years(graduate_df)
undergrad_df    = concatenate_unique_years(undergrad_df)
professional_df = concatenate_unique_years(professional_df)
law_df          = concatenate_unique_years(law_df)
non_cred_df     = concatenate_unique_years(non_cred_df)
scales_df       = concatenate_unique_years(scales_df)


# ---------------------------------------------------------------------------
# 8. TOTAL COMPLETED TERMS
#    Total number of distinct terms ever attended by each student.
# ---------------------------------------------------------------------------
def calculate_number_terms(series: pd.Series) -> list:
    """
    Count the number of comma-separated entries in UNIQUE_TERM_CD.
    Returns a list parallel to the input series.
    """
    return [len(val.split(",")) for val in series]


for seg in [graduate_df, undergrad_df, professional_df,
            law_df, non_cred_df, scales_df]:
    seg["TOTAL_COMPLETED_TERMS"] = calculate_number_terms(seg["UNIQUE_TERM_CD"])


# ---------------------------------------------------------------------------
# 9. CURRENT COMPLETED TERMS
#    For each row (a specific student-term snapshot), this is the sequential
#    rank of that term among all terms the student ever attended, ordered
#    chronologically.  So if a student attended 8 terms total and the current
#    row corresponds to their 4th term, CURRENT_COMPLETED_TERMS = 4.
# ---------------------------------------------------------------------------
def calculate_current_completed_terms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add CURRENT_COMPLETED_TERMS: the ordinal position of each TERM_CD
    within a student's personal chronological term sequence.
    """
    df["CURRENT_COMPLETED_TERMS"] = 1
    unique_ids = df["EDW_PERS_ID"].unique()

    for student_id in unique_ids:
        mask = df["EDW_PERS_ID"] == student_id
        temp_df = df[mask]
        idx_vals = temp_df.index
        terms_sorted = sorted(temp_df["TERM_CD"].unique())
        rank_map = {term: rank for rank, term in enumerate(terms_sorted, start=1)}

        for idx in idx_vals:
            term = df.at[idx, "TERM_CD"]
            if term in rank_map:
                df.at[idx, "CURRENT_COMPLETED_TERMS"] = rank_map[term]

    return df


graduate_df     = calculate_current_completed_terms(graduate_df)
undergrad_df    = calculate_current_completed_terms(undergrad_df)
professional_df = calculate_current_completed_terms(professional_df)
law_df          = calculate_current_completed_terms(law_df)
non_cred_df     = calculate_current_completed_terms(non_cred_df)
scales_df       = calculate_current_completed_terms(scales_df)


# ---------------------------------------------------------------------------
# 10. CURRENT COMPLETED ACADEMIC YEARS
#     Same as above but ranks distinct academic years instead of terms.
# ---------------------------------------------------------------------------
def calculate_completed_academic_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add CURRENT_COMPLETED_ACAD_YEARS: the ordinal position of each
    ACAD_YEAR_CODE within a student's personal chronological year sequence.
    """
    df["ACAD_YEAR_CODE"] = df["ACAD_YEAR_CODE"].astype(int)
    df["CURRENT_COMPLETED_ACAD_YEARS"] = 1
    unique_ids = df["EDW_PERS_ID"].unique()

    for student_id in unique_ids:
        mask = df["EDW_PERS_ID"] == student_id
        temp_df = df[mask]
        idx_vals = temp_df.index
        years_sorted = sorted(temp_df["ACAD_YEAR_CODE"].unique())
        rank_map = {year: rank for rank, year in enumerate(years_sorted, start=1)}

        for idx in idx_vals:
            year = df.at[idx, "ACAD_YEAR_CODE"]
            if year in rank_map:
                df.at[idx, "CURRENT_COMPLETED_ACAD_YEARS"] = rank_map[year]

    return df


graduate_df     = calculate_completed_academic_years(graduate_df)
undergrad_df    = calculate_completed_academic_years(undergrad_df)
professional_df = calculate_completed_academic_years(professional_df)
law_df          = calculate_completed_academic_years(law_df)
non_cred_df     = calculate_completed_academic_years(non_cred_df)
scales_df       = calculate_completed_academic_years(scales_df)


# ---------------------------------------------------------------------------
# 11. EXPORT INTERMEDIATE DATA (BY TERM)
# ---------------------------------------------------------------------------
graduate_df.to_parquet("GRADUATE_BY_TERM.parquet",         engine="pyarrow", compression="snappy")
undergrad_df.to_parquet("UNDERGRADUATE_BY_TERM.parquet",   engine="pyarrow", compression="snappy")
professional_df.to_parquet("PROFESSIONAL_BY_TERM.parquet", engine="pyarrow", compression="snappy")
law_df.to_parquet("LAW_BY_TERM.parquet",                   engine="pyarrow", compression="snappy")
non_cred_df.to_parquet("NON_CRED_BY_TERM.parquet",         engine="pyarrow", compression="snappy")
scales_df.to_parquet("SCALES_BY_TERM.parquet",             engine="pyarrow", compression="snappy")

combined_df = pd.concat(
    [graduate_df, undergrad_df, professional_df, law_df, non_cred_df, scales_df],
    ignore_index=True,
)
combined_df.to_parquet("COMBINED_BY_TERM.parquet", engine="pyarrow", compression="snappy")


# ---------------------------------------------------------------------------
# 12. LABEL TARGET VARIABLES – NEXT TERM ENROLLMENT
#     For each row, determine whether the student enrolled in the
#     *immediately* following term:
#       fall  (8) → next spring (next-year 1)
#       spring(1) → same-year fall (8)
#       summer(5) → same-year fall (8)
#
#     Labels
#       0 – did not enroll (dropped out)
#       1 – enrolled in the next term (or currently in the most-recent year)
#       2 – did not enroll but completed a degree (graduated)
# ---------------------------------------------------------------------------
def label_next_term_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add TARGET_ENROLLED_NEXT_TERM to df.

    Uses UNIQUE_TERM_CD (the full history string) to check whether the
    expected next term appears in the student's record.
    Students in the current academic year are assumed to be continuing (1).
    Graduated students who did not re-enroll are relabeled 2.
    """
    def get_next_term(current_term: str) -> list:
        college  = current_term[0]
        year     = current_term[1:5]
        semester = current_term[5]

        if semester == "8":
            return [college + str(int(year) + 1) + "1"]
        elif semester in ("1", "5"):
            return [college + year + "8"]
        return []

    df.reset_index(drop=True, inplace=True)
    target = []

    for i in range(len(df)):
        next_terms  = get_next_term(str(df.at[i, "TERM_CD"]))
        history     = df.at[i, "UNIQUE_TERM_CD"].split(",")
        target.append("1" if any(t in history for t in next_terms) else "0")

    df["TARGET_ENROLLED_NEXT_TERM"] = target

    for i in range(len(df)):
        # Override: graduated student who did not re-enroll → 2
        if (df.at[i, "STUDENT_AH_DEG_CUR_INFO_IND"] == "Y"
                and df.at[i, "TARGET_ENROLLED_NEXT_TERM"] == "0"):
            df.at[i, "TARGET_ENROLLED_NEXT_TERM"] = "2"
        # Override: current year students are assumed to be continuing
        if df.at[i, "ACAD_YEAR_CODE"] == current_acad_yr_cd_int:
            df.at[i, "TARGET_ENROLLED_NEXT_TERM"] = "1"

    return df


graduate_df     = label_next_term_target_variables(graduate_df)
undergrad_df    = label_next_term_target_variables(undergrad_df)
professional_df = label_next_term_target_variables(professional_df)
law_df          = label_next_term_target_variables(law_df)
non_cred_df     = label_next_term_target_variables(non_cred_df)
scales_df       = label_next_term_target_variables(scales_df)


# ---------------------------------------------------------------------------
# 13. BINARY SUCCESS TARGET – BY TERM
#     Merges "graduated" (2) into "continued" (1) to produce a simpler
#     binary flag: 0 = not successful, 1 = successful.
# ---------------------------------------------------------------------------
def successful_student_target_by_term(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add SUCCESSFUL_OUTCOME_TARGET_BY_TERM: a binary copy of
    TARGET_ENROLLED_NEXT_TERM where label 2 (graduated) is recoded as 1.
    """
    df["SUCCESSFUL_OUTCOME_TARGET_BY_TERM"] = df["TARGET_ENROLLED_NEXT_TERM"].copy()
    df.loc[df["SUCCESSFUL_OUTCOME_TARGET_BY_TERM"] == "2",
           "SUCCESSFUL_OUTCOME_TARGET_BY_TERM"] = "1"
    return df


graduate_df     = successful_student_target_by_term(graduate_df)
undergrad_df    = successful_student_target_by_term(undergrad_df)
professional_df = successful_student_target_by_term(professional_df)
law_df          = successful_student_target_by_term(law_df)
non_cred_df     = successful_student_target_by_term(non_cred_df)
scales_df       = successful_student_target_by_term(scales_df)


# ---------------------------------------------------------------------------
# 14. LABEL TARGET VARIABLES – NEXT YEAR ENROLLMENT (ANY TERM)
#     A student is labeled 1 if they appeared in *any* term during the
#     following academic year (not necessarily the immediately next term).
#     This is more lenient than the next-term check and captures students
#     who took a single semester off.
# ---------------------------------------------------------------------------
def label_next_year_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add TARGET_ENROLLED_NEXT_YEAR to df.

    Checks whether the student attended any term in the next academic year:
      fall  (8) → next year fall/spring/summer
      spring(1) or summer(5) → same-year fall or next-year spring/summer
    """
    def get_next_year_terms(current_term: str) -> list:
        college  = current_term[0]
        year     = current_term[1:5]
        semester = current_term[5]

        if semester == "8":
            y1 = str(int(year) + 1)
            y2 = str(int(year) + 2)
            return [college + y1 + "8", college + y2 + "1", college + y2 + "5"]
        elif semester in ("1", "5"):
            y0 = year
            y1 = str(int(year) + 1)
            return [college + y0 + "8", college + y1 + "1", college + y1 + "5"]
        return []

    df.reset_index(drop=True, inplace=True)
    target = []

    for i in range(len(df)):
        next_terms = get_next_year_terms(str(df.at[i, "TERM_CD"]))
        history    = df.at[i, "UNIQUE_TERM_CD"].split(",")
        target.append("1" if any(t in history for t in next_terms) else "0")

    df["TARGET_ENROLLED_NEXT_YEAR"] = target

    for i in range(len(df)):
        if (df.at[i, "STUDENT_AH_DEG_CUR_INFO_IND"] == "Y"
                and df.at[i, "TARGET_ENROLLED_NEXT_YEAR"] == "0"):
            df.at[i, "TARGET_ENROLLED_NEXT_YEAR"] = "2"
        if df.at[i, "ACAD_YEAR_CODE"] == current_acad_yr_cd_int:
            df.at[i, "TARGET_ENROLLED_NEXT_YEAR"] = "1"

    return df


graduate_df     = label_next_year_target_variables(graduate_df)
undergrad_df    = label_next_year_target_variables(undergrad_df)
professional_df = label_next_year_target_variables(professional_df)
law_df          = label_next_year_target_variables(law_df)
non_cred_df     = label_next_year_target_variables(non_cred_df)
scales_df       = label_next_year_target_variables(scales_df)


# ---------------------------------------------------------------------------
# 15. BINARY SUCCESS TARGET – BY YEAR
# ---------------------------------------------------------------------------
def successful_student_target_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add SUCCESSFUL_OUTCOME_TARGET_BY_YEAR: binary version of
    TARGET_ENROLLED_NEXT_YEAR (2 → 1).
    """
    df["SUCCESSFUL_OUTCOME_TARGET_BY_YEAR"] = df["TARGET_ENROLLED_NEXT_YEAR"].copy()
    df.loc[df["SUCCESSFUL_OUTCOME_TARGET_BY_YEAR"] == "2",
           "SUCCESSFUL_OUTCOME_TARGET_BY_YEAR"] = "1"
    return df


graduate_df     = successful_student_target_by_year(graduate_df)
undergrad_df    = successful_student_target_by_year(undergrad_df)
professional_df = successful_student_target_by_year(professional_df)
law_df          = successful_student_target_by_year(law_df)
non_cred_df     = successful_student_target_by_year(non_cred_df)
scales_df       = successful_student_target_by_year(scales_df)


# ---------------------------------------------------------------------------
# 16. LABEL TARGET VARIABLES – NEXT FALL ENROLLMENT ONLY
#     Stricter variant: counts only next-fall re-enrollment as continuation.
#       fall  (8) → next calendar year's fall
#       spring(1) or summer(5) → same calendar year's fall
# ---------------------------------------------------------------------------
def label_next_fall_year_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add TARGET_ENROLLED_NEXT_FALL_YEAR to df.

    Checks whether the student appears in the single fall term that
    follows the current term's academic year.
    """
    def get_next_fall_term(current_term: str) -> list:
        college  = current_term[0]
        year     = current_term[1:5]
        semester = current_term[5]

        if semester == "8":
            return [college + str(int(year) + 1) + "8"]
        elif semester in ("1", "5"):
            return [college + year + "8"]
        return []

    df.reset_index(drop=True, inplace=True)
    target = []

    for i in range(len(df)):
        next_terms = get_next_fall_term(str(df.at[i, "TERM_CD"]))
        history    = df.at[i, "UNIQUE_TERM_CD"].split(",")
        target.append("1" if any(t in history for t in next_terms) else "0")

    df["TARGET_ENROLLED_NEXT_FALL_YEAR"] = target

    for i in range(len(df)):
        if (df.at[i, "STUDENT_AH_DEG_CUR_INFO_IND"] == "Y"
                and df.at[i, "TARGET_ENROLLED_NEXT_FALL_YEAR"] == "0"):
            df.at[i, "TARGET_ENROLLED_NEXT_FALL_YEAR"] = "2"
        if df.at[i, "ACAD_YEAR_CODE"] == current_acad_yr_cd_int:
            df.at[i, "TARGET_ENROLLED_NEXT_FALL_YEAR"] = "1"

    return df


graduate_df     = label_next_fall_year_target_variables(graduate_df)
undergrad_df    = label_next_fall_year_target_variables(undergrad_df)
professional_df = label_next_fall_year_target_variables(professional_df)
law_df          = label_next_fall_year_target_variables(law_df)
non_cred_df     = label_next_fall_year_target_variables(non_cred_df)
scales_df       = label_next_fall_year_target_variables(scales_df)


# ---------------------------------------------------------------------------
# 17. EXPORT LABELED DATA – ONE ROW PER STUDENT-TERM
# ---------------------------------------------------------------------------
graduate_df.to_parquet("GRADUATE_TARGET_BY_TERM.parquet",         engine="pyarrow", compression="snappy")
undergrad_df.to_parquet("UNDERGRADUATE_TARGET_BY_TERM.parquet",   engine="pyarrow", compression="snappy")
professional_df.to_parquet("PROFESSIONAL_TARGET_BY_TERM.parquet", engine="pyarrow", compression="snappy")
law_df.to_parquet("LAW_TARGET_BY_TERM.parquet",                   engine="pyarrow", compression="snappy")
non_cred_df.to_parquet("NON_CRED_TARGET_BY_TERM.parquet",         engine="pyarrow", compression="snappy")
scales_df.to_parquet("SCALES_TARGET_BY_TERM.parquet",             engine="pyarrow", compression="snappy")


# ---------------------------------------------------------------------------
# 18. AGGREGATE TO ONE ROW PER STUDENT PER ACADEMIC YEAR
#     Retains only the *most recent* semester snapshot for each
#     (student, academic year) pair using a priority mapping:
#       summer (5) → priority 1 (most recent within year)
#       spring (1) → priority 2
#       fall   (8) → priority 3
#     idxmin() selects the row with the lowest priority number (most recent).
# ---------------------------------------------------------------------------
def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce df to one row per (EDW_PERS_ID, ACAD_YEAR_CODE) by keeping
    the most recent semester within each academic year.

    Semester priority (lower = more recent):
        summer (5) → 1
        spring (1) → 2
        fall   (8) → 3
    """
    df["SEMESTER_CD"] = df["SEMESTER_CD"].astype(int)

    # Map semester digit to priority (lower = most recent in the year)
    semester_priority = {1: 2, 5: 1, 8: 3}
    df["SEMESTER_CD"] = df["SEMESTER_CD"].replace(semester_priority)

    # Keep only the row with the minimum priority (most recent semester)
    most_recent_idx = df.groupby(["EDW_PERS_ID", "ACAD_YEAR_CODE"])["SEMESTER_CD"].idxmin()
    return df.loc[most_recent_idx]


graduate_df     = process_dataframe(graduate_df)
undergrad_df    = process_dataframe(undergrad_df)
professional_df = process_dataframe(professional_df)
law_df          = process_dataframe(law_df)
non_cred_df     = process_dataframe(non_cred_df)
scales_df       = process_dataframe(scales_df)


# ---------------------------------------------------------------------------
# 19. EXPORT LABELED DATA – ONE ROW PER STUDENT-YEAR
# ---------------------------------------------------------------------------
graduate_df.to_parquet("GRADUATE_TARGET_BY_YEAR.parquet",         engine="pyarrow", compression="snappy")
undergrad_df.to_parquet("UNDERGRADUATE_TARGET_BY_YEAR.parquet",   engine="pyarrow", compression="snappy")
professional_df.to_parquet("PROFESSIONAL_TARGET_BY_YEAR.parquet", engine="pyarrow", compression="snappy")
law_df.to_parquet("LAW_TARGET_BY_YEAR.parquet",                   engine="pyarrow", compression="snappy")
non_cred_df.to_parquet("NON_CRED_TARGET_BY_YEAR.parquet",         engine="pyarrow", compression="snappy")
scales_df.to_parquet("SCALES_TARGET_BY_YEAR.parquet",             engine="pyarrow", compression="snappy")

combined_df = pd.concat(
    [graduate_df, undergrad_df, professional_df, law_df, non_cred_df, scales_df]
)
combined_df.to_parquet("COMBINED_TARGET_BY_YEAR.parquet", engine="pyarrow", compression="snappy")


# ---------------------------------------------------------------------------
# 20. SINGLE RANDOM SAMPLE PER STUDENT
#     Reduces each segment to one row per EDW_PERS_ID by randomly selecting
#     a single snapshot.  This prevents a single student from appearing with
#     different (potentially contradictory) labels during model training.
# ---------------------------------------------------------------------------
def keep_single_point_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with exactly one row per EDW_PERS_ID, chosen
    randomly (random_state=42 for reproducibility).
    """
    shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return shuffled.groupby("EDW_PERS_ID").first().reset_index()


new_graduate_df     = keep_single_point_data(graduate_df)
new_undergrad_df    = keep_single_point_data(undergrad_df)
new_professional_df = keep_single_point_data(professional_df)
new_law_df          = keep_single_point_data(law_df)
new_non_cred_df     = keep_single_point_data(non_cred_df)
new_scales_df       = keep_single_point_data(scales_df)


# ---------------------------------------------------------------------------
# 21. EXPORT SINGLE-POINT DATA
#     Recommended for model training to avoid having one student mapped to
#     multiple (possibly conflicting) target labels.
# ---------------------------------------------------------------------------
new_graduate_df.to_parquet("GRADUATE_TARGET_BY_YEAR_SINGLE_RANDOM_POINT.parquet",         engine="pyarrow", compression="snappy")
new_undergrad_df.to_parquet("UNDERGRADUATE_TARGET_BY_YEAR_SINGLE_RANDOM_POINT.parquet",   engine="pyarrow", compression="snappy")
new_professional_df.to_parquet("PROFESSIONAL_TARGET_BY_YEAR_SINGLE_RANDOM_POINT.parquet", engine="pyarrow", compression="snappy")
new_law_df.to_parquet("LAW_TARGET_BY_YEAR_SINGLE_RANDOM_POINT.parquet",                   engine="pyarrow", compression="snappy")
new_non_cred_df.to_parquet("NON_CRED_TARGET_BY_YEAR_SINGLE_RANDOM_POINT.parquet",         engine="pyarrow", compression="snappy")
new_scales_df.to_parquet("SCALES_TARGET_BY_YEAR_SINGLE_RANDOM_POINT.parquet",             engine="pyarrow", compression="snappy")

combined_single_df = pd.concat(
    [new_graduate_df, new_undergrad_df, new_professional_df,
     new_law_df, new_non_cred_df, new_scales_df],
    ignore_index=True,
)
combined_single_df.to_parquet(
    "COMBINED_TARGET_BY_YEAR_SINGLE_RANDOM_POINT.parquet",
    engine="pyarrow",
    compression="snappy",
)
