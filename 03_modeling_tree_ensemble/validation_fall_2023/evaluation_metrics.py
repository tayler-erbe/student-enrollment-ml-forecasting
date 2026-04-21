"""
evaluation_metrics.py
----------------------
Evaluates the performance of the Random Forest (RF) classification model that
predicts whether continuing students will re-enroll in the upcoming Fall semester.

The script is organized into three stages:

  1. OVERALL MODEL PERFORMANCE
     Compares RF predictions against actual Fall 2023 enrollment for all
     continuing students, producing a single set of aggregate metrics.

  2. PERFORMANCE BY COLLEGE AND STUDENT LEVEL (grouped)
     Breaks the same predictions down by ACAD_COLL_NAME and STUDENT_LEVEL_DESC
     so that weaker subgroups can be identified and targeted for improvement.

  3. PERFORMANCE BY STUDENT LEVEL × COLLEGE (cross-grouped)
     Further stratifies results by every (student level, college) combination,
     enabling a finer-grained assessment across program types.

Input files
-----------
- T_RS_STUDENT_2023.parquet
    Census-snapshot student registration table for AY 2022-23 and Fall 2023.
- COMBINED_TARGET_BY_YEAR.parquet
    Historical enrollment with degree-status labels used to identify graduates.
- EDW_PERS_ID_Students_to_Discontinue.xlsx
    RF model output: students predicted to NOT re-enroll, one sheet per
    student-level group (All, Undergrads, Grads, Professional, Law).

Output files
------------
- Evaluation_Metrics_Fall_2023.xlsx
    College-level and student-level metric summary, sorted by accuracy.
- Evaluation_Metrics_StudentLevel_by_College_Fall_2023.xlsx
    (Student level × College) metric summary, sorted by accuracy.

Metrics reported per group
--------------------------
  Confusion Matrix, Accuracy, Precision, Recall (Sensitivity), F1 Score,
  True Positive, False Positive, True Negative, False Negative
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# File paths  –– update these to match your environment
# ---------------------------------------------------------------------------
STUDENT_FILE_PATH      = r"T_RS_STUDENT_2023.parquet"
COMBINED_TARGET_PATH   = r"COMBINED_TARGET_BY_YEAR.parquet"
DISCONTINUE_EXCEL_PATH = r"EDW_PERS_ID_Students_to_Discontinue.xlsx"
OUTPUT_METRICS_COLLEGE = r"Evaluation_Metrics_Fall_2023.xlsx"
OUTPUT_METRICS_LEVEL   = r"Evaluation_Metrics_StudentLevel_by_College_Fall_2023.xlsx"

# Term codes used throughout the script
FALL_2023_TERM      = "220238"   # Fall 2023 census snapshot
AY_2223_TERMS       = ["220231", "220235", "220228"]  # Spring/Summer/Fall 2022-23

# Sheet names in the discontinue predictions Excel file
SHEET_NAMES = [
    "Pivot_All_Discontinued",
    "Pivot_Undergrads",
    "Pivot_Grads",
    "Pivot_Professional",
    "Pivot_Law",
    "All_Discontinued",
    "Undergrads_Discontinued",
    "Grads_Discontinued",
    "Professional_Discontinued",
    "Law_Discontinued",
]


# ---------------------------------------------------------------------------
# Helper: calculate classification metrics for a single group
# ---------------------------------------------------------------------------
def calculate_metrics(df: pd.DataFrame, label_name: str) -> pd.DataFrame:
    """
    Compute standard binary classification metrics for one subgroup.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'ACTUAL_CONTINUED' and 'RF_PREDICTION_TO_CONTINUE'
        (both encoded as 1 = continues, 0 = does not continue).
    label_name : str
        Human-readable identifier for this group (used in the output table).

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with all metrics plus the label name.

    Notes
    -----
    When sklearn's confusion_matrix produces a 1×1 matrix (only one class
    present in the group), the function promotes it to a 2×2 matrix so that
    ravel() can always unpack four values safely.
    """
    actual    = df["ACTUAL_CONTINUED"]
    predicted = df["RF_PREDICTION_TO_CONTINUE"]

    conf_matrix = confusion_matrix(actual, predicted)

    # Handle edge case: group contains only one class (e.g., all continued)
    if conf_matrix.shape == (1, 1):
        conf_matrix = np.array([[0, 0], [0, conf_matrix[0, 0]]])

    tn, fp, fn, tp = conf_matrix.ravel()

    return pd.DataFrame({
        "Confusion_Matrix": [conf_matrix],
        "Accuracy":         [accuracy_score(actual, predicted)],
        "Precision":        [precision_score(actual, predicted, zero_division=0)],
        "Recall":           [recall_score(actual, predicted, zero_division=0)],
        "F1 Score":         [f1_score(actual, predicted, zero_division=0)],
        "True Positive":    [tp],
        "False Positive":   [fp],
        "True Negative":    [tn],
        "False Negative":   [fn],
        "Label Name":       [label_name],
    })


# ---------------------------------------------------------------------------
# Stage 1 — Load data and build the master predictions DataFrame
# ---------------------------------------------------------------------------
def build_predictions_df() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all inputs and return:
      - df_predictions  : one row per unique student in AY 2022-23,
                          with RF prediction and actual Fall 2023 outcome.
      - df_2023_filtered: AY 2022-23 student records deduplicated to each
                          student's most recent term (for attribute joins).
    """

    # ------------------------------------------------------------------
    # 1a. Load Fall 2023 registrations — used as the ground-truth label
    # ------------------------------------------------------------------
    print("Loading student registration data …")
    df_raw = pd.read_parquet(STUDENT_FILE_PATH)

    # Isolate continuing students registered at the Fall 2023 census
    df_fall = df_raw[
        (df_raw["TERM_CD"] == FALL_2023_TERM) &
        (df_raw["STUDENT_TYPE_CD"] == "C")
    ].copy()

    # Clean EDW_PERS_ID: drop duplicates and normalise to integer
    df_fall["EDW_PERS_ID"] = (
        df_fall["EDW_PERS_ID"]
        .drop_duplicates()
        .astype(str)
        .str.replace(".0", "", regex=False)
        .astype(int)
    )

    # Build the set of students who actually continued into Fall 2023
    enrolled_continued_fall = set(df_fall["EDW_PERS_ID"])

    # ------------------------------------------------------------------
    # 1b. Load combined target file — used to identify graduates
    # ------------------------------------------------------------------
    print("Loading combined target / degree data …")
    original_snapshot = pd.read_parquet(COMBINED_TARGET_PATH)

    # Filter to AY 2022-23 terms
    snapshot_2223 = original_snapshot[
        original_snapshot["TERM_CD"].isin(AY_2223_TERMS)
    ]

    # Students who were awarded a degree during AY 2022-23 are excluded
    # from evaluation (they are expected NOT to re-enroll)
    awarded_df = snapshot_2223[
        snapshot_2223["DEG_STATUS_CD"].astype(str).eq("AW")
    ]
    students_to_graduate = set(awarded_df["EDW_PERS_ID"])

    # ------------------------------------------------------------------
    # 1c. Build the AY 2022-23 student population
    # ------------------------------------------------------------------
    # Normalise EDW_PERS_ID across the full dataset
    df_raw["EDW_PERS_ID"] = (
        df_raw["EDW_PERS_ID"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .astype(int)
    )

    # All students appearing in any AY 2022-23 term
    df_2023 = df_raw[df_raw["TERM_CD"].isin(AY_2223_TERMS)].copy()
    students_2223 = set(df_2023["EDW_PERS_ID"])

    # For attribute joins later, keep each student's most recent term record
    df_2023["TERM_CD"] = df_2023["TERM_CD"].astype(int)
    most_recent_idx = df_2023.groupby("EDW_PERS_ID")["TERM_CD"].idxmax()
    df_2023_filtered = df_2023.loc[most_recent_idx]

    # ------------------------------------------------------------------
    # 1d. Load RF predictions from Excel
    # ------------------------------------------------------------------
    print("Loading RF prediction file …")
    ids_discontinued_dfs = pd.read_excel(DISCONTINUE_EXCEL_PATH, sheet_name=None)

    # The "All_Discontinued" sheet contains the full set of predicted
    # non-continuers across all student levels
    all_discontinued_df = ids_discontinued_dfs.get("All_Discontinued")
    pred_discontinued = set(all_discontinued_df["EDW_PERS_ID"])

    # ------------------------------------------------------------------
    # 1e. Assemble the predictions DataFrame
    # ------------------------------------------------------------------
    df_predictions = pd.DataFrame({"EDW_PERS_ID": list(students_2223)})

    # Remove students expected to graduate — they are out of scope
    df_predictions = df_predictions[
        ~df_predictions["EDW_PERS_ID"].isin(students_to_graduate)
    ].copy()

    # RF prediction: 0 = predicted to discontinue, 1 = predicted to continue
    df_predictions["RF_PREDICTION_TO_CONTINUE"] = df_predictions["EDW_PERS_ID"].apply(
        lambda x: 0 if x in pred_discontinued else 1
    )

    # Actual outcome: 1 = enrolled in Fall 2023, 0 = did not
    df_predictions["ACTUAL_CONTINUED"] = df_predictions["EDW_PERS_ID"].apply(
        lambda x: 1 if x in enrolled_continued_fall else 0
    )

    # ------------------------------------------------------------------
    # 1f. Validation labels (for human-readable review)
    # ------------------------------------------------------------------
    validation_labels = []
    is_correct_flags  = []

    for _, row in df_predictions.iterrows():
        pred   = row["RF_PREDICTION_TO_CONTINUE"]
        actual = row["ACTUAL_CONTINUED"]

        if pred == 1 and actual == 1:
            validation_labels.append("true continued")
            is_correct_flags.append(1)
        elif pred == 0 and actual == 0:
            validation_labels.append("true discontinued")
            is_correct_flags.append(1)
        elif pred == 0 and actual == 1:
            validation_labels.append("false continued")   # model missed a continuer
            is_correct_flags.append(0)
        elif pred == 1 and actual == 0:
            validation_labels.append("false discontinued")  # model missed a leaver
            is_correct_flags.append(0)

    df_predictions["Validation"] = validation_labels
    df_predictions["Is_Correct"]  = is_correct_flags

    return df_predictions, df_2023_filtered


# ---------------------------------------------------------------------------
# Stage 2 — Overall aggregate metrics
# ---------------------------------------------------------------------------
def evaluate_overall(df_predictions: pd.DataFrame) -> None:
    """
    Print the aggregate confusion matrix and scalar metrics for the full
    student population (all levels and colleges combined).
    """
    actual    = df_predictions["ACTUAL_CONTINUED"]
    predicted = df_predictions["RF_PREDICTION_TO_CONTINUE"]

    conf_matrix = confusion_matrix(actual, predicted)
    tn, fp, fn, tp = conf_matrix.ravel()

    accuracy  = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted)
    recall    = recall_score(actual, predicted)
    f1        = f1_score(actual, predicted)

    print("\n=== Overall Model Performance ===")
    print(f"  True Positives  (TP): {tp}")
    print(f"  True Negatives  (TN): {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  Accuracy  : {accuracy:.6f}")
    print(f"  Precision : {precision:.6f}")
    print(f"  Recall    : {recall:.6f}")
    print(f"  F1 Score  : {f1:.6f}")


# ---------------------------------------------------------------------------
# Stage 3 — Metrics grouped by college and by student level
# ---------------------------------------------------------------------------
def evaluate_by_group(
    df_predictions: pd.DataFrame,
    df_2023_filtered: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute metrics for every unique value in ACAD_COLL_NAME and
    STUDENT_LEVEL_DESC, then return a combined DataFrame sorted by accuracy.

    The function joins df_2023_filtered onto df_predictions to bring in the
    STUDENT_LEVEL_DESC and ACAD_COLL_NAME columns, which are not part of
    the original predictions table.
    """

    # Drop duplicate student IDs before joining so each student maps to
    # exactly one set of attributes
    df_unique = df_predictions.drop_duplicates(subset="EDW_PERS_ID")

    # Left-join to attach college and level attributes
    merged_df = df_unique.merge(
        df_2023_filtered[["EDW_PERS_ID", "STUDENT_LEVEL_DESC", "ACAD_COLL_NAME"]],
        on="EDW_PERS_ID",
        how="left",
    )

    grouping_columns = ["STUDENT_LEVEL_DESC", "ACAD_COLL_NAME"]
    result_frames = []

    # Iterate over each grouping dimension and each unique value within it
    for col in grouping_columns:
        for val in merged_df[col].dropna().unique():
            subset = merged_df[merged_df[col] == val]
            result_frames.append(calculate_metrics(subset, f"{col}_{val}"))

    result = (
        pd.concat(result_frames, axis=0, ignore_index=True)
        .sort_values("Accuracy", ascending=False)
        .reset_index(drop=True)
    )

    return result, merged_df   # return merged_df for the next stage


# ---------------------------------------------------------------------------
# Stage 4 — Metrics by (student level × college) cross-group
# ---------------------------------------------------------------------------
def evaluate_by_level_and_college(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Stratify predictions by every (STUDENT_LEVEL_DESC, ACAD_COLL_NAME) pair
    and compute metrics for each cell.  Groups with fewer than ~5 students
    will naturally show unreliable metrics (noted in the assessment comments).
    """
    student_levels = merged_df["STUDENT_LEVEL_DESC"].dropna().unique()
    metrics_dict   = {}

    for level in student_levels:
        level_df    = merged_df[merged_df["STUDENT_LEVEL_DESC"] == level]
        coll_names  = level_df["ACAD_COLL_NAME"].dropna().unique()

        for coll in coll_names:
            subset = level_df[level_df["ACAD_COLL_NAME"] == coll]
            key    = f"{level}_{coll}"
            metrics_dict[key] = calculate_metrics(subset, key)

    result = (
        pd.concat(metrics_dict.values(), axis=0, ignore_index=True)
        .sort_values("Accuracy", ascending=False)
        .reset_index(drop=True)
    )

    return result


# ---------------------------------------------------------------------------
# Convenience filters — slice results by student-level prefix
# ---------------------------------------------------------------------------
def filter_by_level_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Return rows whose Label Name starts with the given student-level prefix."""
    return df[df["Label Name"].str.startswith(prefix)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
def main():

    # Stage 1: load data and build predictions table
    df_predictions, df_2023_filtered = build_predictions_df()

    # Stage 2: print overall aggregate metrics
    evaluate_overall(df_predictions)

    # Stage 3: metrics broken down by college and by student level
    print("\nCalculating metrics by college and student level …")
    grouped_result, merged_df = evaluate_by_group(df_predictions, df_2023_filtered)

    print(f"Saving group-level metrics → {OUTPUT_METRICS_COLLEGE}")
    grouped_result.to_excel(OUTPUT_METRICS_COLLEGE, index=False)

    # Stage 4: metrics broken down by (student level × college) cross-group
    print("Calculating metrics by student level × college …")
    level_college_result = evaluate_by_level_and_college(merged_df)

    print(f"Saving level×college metrics → {OUTPUT_METRICS_LEVEL}")
    level_college_result.to_excel(OUTPUT_METRICS_LEVEL, index=False)

    # Optional: print sub-tables for quick review
    for prefix in ["Undergrad", "Graduate", "Professional", "Law", "Non-Credit"]:
        subset = filter_by_level_prefix(level_college_result, prefix)
        print(f"\n--- {prefix} ({len(subset)} groups) ---")
        print(subset[["Label Name", "Accuracy", "Precision", "Recall", "F1 Score"]].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
