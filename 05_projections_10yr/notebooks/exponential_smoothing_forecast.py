"""
exponential_smoothing_forecast.py
----------------------------------
Fits Holt's Double Exponential Smoothing (additive trend, no seasonality) models
to continuing-student enrollment data for colleges that had poor-performing ARIMA
models.  For each college / student-level combination the script:
  1. Loads historical enrollment from a base Parquet file.
  2. Splits the series 80/20 into train and validation sets.
  3. Grid-searches over alpha (level) and beta (trend) smoothing parameters to
     minimise validation RMSE.
  4. Re-fits on the full series with the best parameters.
  5. Produces a 10-period point forecast plus a MAE-based confidence band.
  6. Saves the combined results to both Parquet and Excel.

Input files
-----------
- ARIMA_Continuing_Knime/10YEAR_CombinedStatsForecasts-ContinuingStudents_Poor_Performing_R2.xlsx
    Rows identify (College Name, Student Level) pairs whose ARIMA models
    underperformed and should be replaced with exponential smoothing.
- ARIMA_Continuing_Knime/BaseFiles.parquet
    Long-format enrollment table with at least the columns:
        ACAD_COLL_NAME, COLL_CD, TERM_CD, STUDENT_LEVEL_DESC, Continuing

Output files
------------
- ARIMA_Continuing_Knime/Exponential_Smoothing_data.parquet
- ARIMA_Continuing_Knime/Exponential_Smoothing_data.xlsx
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.api import ExponentialSmoothing

# ---------------------------------------------------------------------------
# File paths  –– update these to match your environment
# ---------------------------------------------------------------------------
POOR_MODELS_PATH = r"ARIMA_Continuing_Knime/10YEAR_CombinedStatsForecasts-ContinuingStudents_Poor_Performing_R2.xlsx"
BASE_FILE_PATH   = r"ARIMA_Continuing_Knime/BaseFiles.parquet"
OUTPUT_PARQUET   = r"ARIMA_Continuing_Knime/Exponential_Smoothing_data.parquet"
OUTPUT_EXCEL     = r"ARIMA_Continuing_Knime/Exponential_Smoothing_data.xlsx"

# ---------------------------------------------------------------------------
# Student-level mapping
# ---------------------------------------------------------------------------
# Maps the raw STUDENT_LEVEL_DESC values in the base file to four broad
# categories used throughout the forecasting pipeline.
LEVEL_DICT = {
    "Graduate":      ["Graduate - Chicago",
                      "Graduate Non-Degree Chicago",
                      "Graduate Online – Chicago"],
    "Professional":  ["Professional - Chicago"],
    "Undergraduate": ["Undergrad - Chicago",
                      "Undergrad Non-Degree Chicago"],
    "Other":         ["Non-Credit - Chicago",
                      "SCALES - Chicago"],
}

# Flatten to a simple lookup: raw description → category label
LEVEL_MAP = {desc: key for key, descs in LEVEL_DICT.items() for desc in descs}


def get_student_level(desc: str) -> str:
    """Return the broad student-level category for a raw description string."""
    return LEVEL_MAP.get(desc, "Unknown")


# ---------------------------------------------------------------------------
# Grid-search parameter space
# ---------------------------------------------------------------------------
# alpha controls the smoothing of the level component (0 = no update, 1 = naive).
# beta  controls the smoothing of the trend component.
# Both are swept in steps of 0.05 across (0.05, 0.95].
ALPHAS = np.arange(0.05, 1.0, 0.05)
BETAS  = np.arange(0.05, 1.0, 0.05)


# ---------------------------------------------------------------------------
# Core forecasting function
# ---------------------------------------------------------------------------
def get_arima_dfs(
    enrollments: np.ndarray,
    term_cd_semester: str,
    college_name: str,
    student_level: str,
    term: str,
) -> pd.DataFrame:
    """
    Fit Holt's Double Exponential Smoothing to an enrollment series and return
    a tidy DataFrame of 10-period forecasts.

    Parameters
    ----------
    enrollments : np.ndarray
        Historical enrollment counts ordered chronologically.
    term_cd_semester : str
        Single-character semester code embedded in TERM_CD:
          "8" → Fall, "1" → Spring, "5" → Summer.
    college_name : str
        Academic college label (used for output tagging only).
    student_level : str
        Broad student-level category (used for output tagging only).
    term : str
        Human-readable term name, e.g. "Fall", "Spring", "Summer".

    Returns
    -------
    pd.DataFrame
        One row per forecast horizon (10 rows) with columns:
        Best_Alpha, Best_Beta, Best_RMSE, MAE,
        Forecast_Point_Estimate, Forecast_Range_0, Forecast_Range_1,
        Term_Cd, College_Name, Student_Level, Term.
    """

    # ------------------------------------------------------------------
    # 1. Train / validation split  (80 % train, 20 % validation)
    # ------------------------------------------------------------------
    train_size = int(0.8 * len(enrollments))
    train_data = enrollments[:train_size]
    val_data   = enrollments[train_size:]

    # ------------------------------------------------------------------
    # 2. Grid search over (alpha, beta) to minimise validation RMSE
    # ------------------------------------------------------------------
    best_alpha, best_beta = None, None
    best_rmse = float("inf")

    for alpha in ALPHAS:
        for beta in BETAS:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                model = ExponentialSmoothing(
                    train_data,
                    trend="add",
                    seasonal=None,
                    initialization_method="legacy-heuristic",
                )
                fit = model.fit(
                    smoothing_level=alpha,
                    smoothing_trend=beta,
                    optimized=False,
                )

            forecast_val = fit.forecast(steps=len(val_data))
            rmse = np.sqrt(mean_squared_error(val_data, forecast_val))

            if rmse < best_rmse:
                best_alpha, best_beta = alpha, beta
                best_rmse = rmse

    # ------------------------------------------------------------------
    # 3. Refit on the full series using the best (alpha, beta)
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        best_model = ExponentialSmoothing(
            enrollments,
            trend="add",
            seasonal=None,
            initialization_method="legacy-heuristic",
            use_boxcox=False,
        )
        best_fit = best_model.fit(
            smoothing_level=best_alpha,
            smoothing_trend=best_beta,
            optimized=False,
        )

    # ------------------------------------------------------------------
    # 4. Produce 10-period point forecast and MAE-based confidence band
    # ------------------------------------------------------------------
    point_forecast = best_fit.forecast(steps=10)

    # MAE is computed on the validation window of the full-series model
    mae = mean_absolute_error(val_data, best_fit.forecast(steps=len(val_data)))

    lower_bound = point_forecast - mae
    upper_bound = point_forecast + mae

    # ------------------------------------------------------------------
    # 5. Build future TERM_CD codes starting from the next cycle
    # ------------------------------------------------------------------
    current_year = datetime.now().year
    term_code_list = []

    if term_cd_semester == "8":       # Fall: first forecast year is current_year
        start_year = current_year - 1
        for i in range(1, 11):
            term_code_list.append(f"2{start_year + i}8")

    elif term_cd_semester == "1":     # Spring
        start_year = current_year
        for i in range(1, 11):
            term_code_list.append(f"2{start_year + i}1")

    elif term_cd_semester == "5":     # Summer
        start_year = current_year
        for i in range(1, 11):
            term_code_list.append(f"2{start_year + i}5")

    # ------------------------------------------------------------------
    # 6. Assemble output DataFrame
    # ------------------------------------------------------------------
    n = len(point_forecast)
    df = pd.DataFrame({
        "Best_Alpha":              [best_alpha] * n,
        "Best_Beta":               [best_beta]  * n,
        "Best_RMSE":               [best_rmse]  * n,
        "MAE":                     [mae]         * n,
        "Forecast_Point_Estimate": point_forecast,
        "Forecast_Range_0":        list(lower_bound),
        "Forecast_Range_1":        list(upper_bound),
        "Term_Cd":                 term_code_list,
        "College_Name":            [college_name]   * n,
        "Student_Level":           [student_level]  * n,
        "Term":                    [term]           * n,
    })

    return df


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # Load input data
    # ------------------------------------------------------------------
    print("Loading poor-performing model list …")
    poor_performing_models = pd.read_excel(POOR_MODELS_PATH, engine="openpyxl")

    print("Loading base enrollment file …")
    base_df = pd.read_parquet(BASE_FILE_PATH)

    # Map raw level descriptions to broad categories
    base_df["Student Level"] = base_df["STUDENT_LEVEL_DESC"].apply(get_student_level)

    # ------------------------------------------------------------------
    # Loop over each poor-performing (college, student level) pair and
    # fit exponential smoothing models for Fall, Spring, and Summer terms
    # ------------------------------------------------------------------
    fall_arima   = {}
    spring_arima = {}
    summer_arima = {}

    for i in range(len(poor_performing_models)):
        college = poor_performing_models.loc[i, "College Name"]
        level   = poor_performing_models.loc[i, "Student Level"]
        print(f"  Processing row {i}: {college} | {level}")

        # Common filter mask (college + student level)
        mask = (
            (base_df["ACAD_COLL_NAME"] == college) &
            (base_df["Student Level"]  == level)
        )

        # ---- Fall (semester code "8") ---------------------------------
        fall_df = (
            base_df[mask & (base_df["TERM_CD"].str[5:6] == "8")]
            .groupby(["ACAD_COLL_NAME", "COLL_CD", "TERM_CD", "Student Level"])["Continuing"]
            .sum()
            .reset_index()
        )
        fall_enrollments = fall_df["Continuing"].to_numpy()
        fall_arima[i] = get_arima_dfs(
            fall_enrollments, "8",
            fall_df["ACAD_COLL_NAME"].iloc[0],
            fall_df["Student Level"].iloc[0],
            "Fall",
        )

        # ---- Spring (semester code "1") -------------------------------
        spring_df = (
            base_df[mask & (base_df["TERM_CD"].str[5:6] == "1")]
            .groupby(["ACAD_COLL_NAME", "COLL_CD", "TERM_CD", "Student Level"])["Continuing"]
            .sum()
            .reset_index()
        )
        spring_enrollments = spring_df["Continuing"].to_numpy()
        spring_arima[i] = get_arima_dfs(
            spring_enrollments, "1",
            spring_df["ACAD_COLL_NAME"].iloc[0],
            spring_df["Student Level"].iloc[0],
            "Spring",
        )

        # ---- Summer (semester code "5") -------------------------------
        summer_df = (
            base_df[mask & (base_df["TERM_CD"].str[5:6] == "5")]
            .groupby(["ACAD_COLL_NAME", "COLL_CD", "TERM_CD", "Student Level"])["Continuing"]
            .sum()
            .reset_index()
        )
        summer_enrollments = summer_df["Continuing"].to_numpy()
        summer_arima[i] = get_arima_dfs(
            summer_enrollments, "5",
            summer_df["ACAD_COLL_NAME"].iloc[0],
            summer_df["Student Level"].iloc[0],
            "Summer",
        )

    # ------------------------------------------------------------------
    # Combine all term forecasts into a single DataFrame
    # ------------------------------------------------------------------
    print("Concatenating results …")
    arima_data = pd.concat(
        [
            pd.concat(fall_arima.values(),   ignore_index=True),
            pd.concat(spring_arima.values(), ignore_index=True),
            pd.concat(summer_arima.values(), ignore_index=True),
        ],
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    print(f"Saving Parquet → {OUTPUT_PARQUET}")
    arima_data.to_parquet(OUTPUT_PARQUET)

    print(f"Saving Excel   → {OUTPUT_EXCEL}")
    arima_data.to_excel(OUTPUT_EXCEL, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
