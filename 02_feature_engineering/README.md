# README: Python Data Preparation and Feature Engineering
**UIC Student Enrollment Prediction Project — 2023**

---

## Overview

This module handles all data preparation, cleaning, and feature engineering steps applied to the raw student enrollment data prior to model training. The inputs are sourced from the Knime data wrangling workflow (see the companion Knime README for query and join details). The outputs are a series of Parquet files ready for use in downstream ARIMA, Gradient Boosted Tree, and classification models.

---

## Files in This Module

| File | Description |
|---|---|
| `data_preparation_feature_engineering.py` | Main Python script for all cleaning and feature engineering |
| `BaseFiles.parquet` | Raw data from Knime workflow (AY 1213–2223) |
| `featured_data_YEAR.parquet` | Processed data with engineered features, labeled target by **YEAR** |
| `featured_data_TERM.parquet` | Processed data with engineered features, labeled target by **TERM** |
| `featured_data_YEAR_deduped.parquet` | Year-labeled data with one randomly sampled record per `edw_pers_id` to prevent overfitting |

---

## Source Data

Raw data is pulled from the Oracle EDW schema (`DSPROD01`) via Knime and covers census-snapshot enrollment records from **Fall 2013 through Summer 2023**. Key source tables include:

- `EDW.T_RS_PERS` — Demographics
- `EDW.T_RS_STUDENT` — Academic standing, level, college, major
- `EDW.T_RS_STUDENT_HOLD` — Registration, transcript, and graduation holds
- `EDW.T_STUDENT_TERM` — Leave of absence information
- `EDW.T_STUDENT_AH_DEG_HIST` — Degree history and graduation status
- `EDW.T_RS_MAX_TEST_SCORE` — ACT/SAT test score presence
- `EDW.t_rs_student_attr` — Special designations (first-gen, honors, etc.)
- `EDW.t_rs_student_ah_level_gpa` — Transfer and institutional GPA
- `EDW.V_ADM_STD_HS_HIST` — High school GPA
- `EDW.t_rs_addr` — Geographic/address information
- Placement exam scores for Math, English, and Chemistry

---

## Data Cleaning Steps

1. **Duplicate handling** — Multiple records per `edw_pers_id` can exist across terms. Depending on the target type (YEAR vs. TERM), duplicates are handled differently (see Deduplication section below).
2. **Missing value treatment**
   - ACT/SAT scores: converted to binary presence indicator (`has_test_score`: 0/1) due to high missingness after scores became optional.
   - Placement scores: imputed with median or flagged with a missing indicator where appropriate.
   - High school GPA: rows with no GPA record are flagged with a binary `has_hsgpa` indicator.
   - Holds: null hold fields treated as no hold present (0).
3. **Filtering** — Only active, census-snapshot, registered students at campus code `200` (Chicago) are retained.
4. **Type casting** — Term codes, academic year codes, and categorical fields are cast to consistent types.

---

## Feature Engineering

### Demographic Features
| Feature | Description |
|---|---|
| `sex_cd` | Encoded sex code |
| `race_eth_group` | Grouped IPEDS race/ethnicity category |
| `age_at_census` | Student age at census date |
| `citizenship_group` | Grouped citizenship type |
| `marital_status` | Encoded marital status |

### Academic Features
| Feature | Description |
|---|---|
| `student_level_group` | Broad level (Undergraduate, Graduate, Professional, Other) |
| `acad_coll_name` | Academic college |
| `student_type_cd` | Student type (e.g., first-time freshman, transfer) |
| `calc_cls_desc` | Calculated class standing (Freshman, Sophomore, etc.) |
| `student_res_group` | Residency group (in-state, out-of-state, international) |
| `student_educ_goal_desc` | Stated educational goal |
| `level_credit_group` | Credit level group |
| `admit_term_cd` | Term of initial admission |
| `expected_grad_term_cd` | Expected graduation term |

### GPA Features
| Feature | Description |
|---|---|
| `transfer_gpa` | Transfer GPA (from level GPA table) |
| `institutional_gpa` | Institutional GPA |
| `hsgpa` | High school GPA (from admissions history) |
| `has_hsgpa` | Binary indicator: whether a HS GPA record exists |

### Test Score Features
| Feature | Description |
|---|---|
| `has_test_score` | Binary: whether ACT or SAT score is on record (scores themselves are excluded due to low coverage post-2020) |
| `act_composite` | ACT composite score (where available) |
| `placement_math` | Math placement exam score |
| `placement_english` | English placement exam score |
| `placement_chemistry` | Chemistry placement exam score |

### Hold Features
| Feature | Description |
|---|---|
| `has_reg_hold` | Binary: registration hold present at census |
| `has_transcript_hold` | Binary: transcript hold present |
| `has_grad_hold` | Binary: graduation hold present |
| `hold_owe_amt` | Dollar amount owed on any financial hold |

### Student Status & LOA Features
| Feature | Description |
|---|---|
| `student_status_cd` | Enrollment status code |
| `has_loa` | Binary: student was on leave of absence during the period |

### Attribute / Program Features
| Feature | Description |
|---|---|
| `is_first_gen` | Binary: first-generation student flag (from attributes table, code `2SES` or `2SEG`) |
| `is_honors` | Binary: enrolled in an honors program |
| `has_special_aid_program` | Binary: associated with a program carrying special financial aid designation |

### Geographic Features
| Feature | Description |
|---|---|
| `addr_state_cd` | State of student's address |
| `addr_zip_cd` | ZIP code |
| `addr_county_name` | County of residence |
| `hs_state_cd` | State of high school attended (undergrad only) |
| `hs_city` | City of high school attended |

### Degree History Features
| Feature | Description |
|---|---|
| `deg_level_cd` | Degree level code (bachelor's, master's, doctoral, etc.) |
| `award_catgry_cd` | Award category |
| `has_prior_degree` | Binary: student has a prior completed degree on record |

### Time / Term Features
| Feature | Description |
|---|---|
| `term_cd` | Census term code |
| `acad_yr_cd` | Academic year code |
| `term_semester` | Extracted semester: Fall (8), Spring (1), Summer (5) |
| `years_since_admit` | Difference between current term year and admit term year |

---

## Target Variable Definitions

### Year-Based Target (`featured_data_YEAR.parquet`)
The target is defined at the **academic year** level. A student is labeled based on whether they were enrolled in the subsequent academic year. Possible labels:

| Label | Description |
|---|---|
| `Continuing` | Student re-enrolled in the next academic year |
| `Graduated` | Student completed their degree (flagged via `STUDENT_AH_DEG_CUR_INFO_IND`) |
| `Stopped Out` | Student did not re-enroll and did not graduate |

### Term-Based Target (`featured_data_TERM.parquet`)
The same label logic applied at the **term** level (Fall → Spring, Spring → Summer, Summer → Fall).

---

## Deduplication (`featured_data_YEAR_deduped.parquet`)

Because a single student (`edw_pers_id`) can appear in multiple terms within an academic year, including all records risks overfitting classifiers to students with many observations. In the deduplicated file, **one record per `edw_pers_id` per academic year is randomly selected** using a fixed random seed for reproducibility. This file is the recommended input for training classification models.

---

## Aggregation Notes

- **Holds**: If multiple hold records exist for a student in a term, they are aggregated by taking the maximum of each binary indicator and the sum of `hold_owe_amt`.
- **Attributes**: Multiple attribute codes are pivoted into individual binary columns. Only codes with meaningful correlation to the target were retained (see code comments for full list).
- **Placement scores**: Where a student has multiple attempts, the maximum score per subject is used.
- **GPA**: Transfer and institutional GPA are taken from the most recent available record.

---

## Known Limitations & Future Work

- **Financial aid data was excluded** from the current model due to data use restrictions. The Knime workflow retains the relevant join logic in a separate section for future reintegration pending approval.
- **SAT scores** have extensive missingness post-2020 (scores became optional). The presence indicator is a reasonable proxy, but adding imputed scores or SAT-optional flags could improve signal.
- **Athlete and advisor tables** were excluded due to high missingness (>60%). These may be worth revisiting if data quality improves.
- **Geographic features** such as ZIP code and county are currently used as-is. Enriching these with census socioeconomic indicators (e.g., median household income, urban/rural classification) is recommended for future iterations.
- The random seed for deduplication is set in the script; document this value if results need to be reproduced exactly.

---

*Last updated: 2023 | Maintained by: Tayler*
