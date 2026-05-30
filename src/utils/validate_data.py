import pandas as pd
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset.

    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    """
    print("🔍 Starting data validation...")

    # === FIX: Convert numeric columns that may be stored as strings ===
    for col in ["TotalCharges", "MonthlyCharges", "tenure"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    failed_expectations = []
    total_checks = 0

    def check(condition, message):
        nonlocal total_checks
        total_checks += 1
        if not condition:
            failed_expectations.append(message)

    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    print("   📋 Validating schema and required columns...")

    required_columns = [
        "customerID", "gender", "Partner", "Dependents",
        "PhoneService", "InternetService", "Contract",
        "tenure", "MonthlyCharges", "TotalCharges"
    ]
    for col in required_columns:
        check(col in df.columns, f"expect_column_to_exist: {col}")

    # Null checks for critical columns
    for col in ["customerID", "tenure", "MonthlyCharges"]:
        if col in df.columns:
            check(df[col].isnull().sum() == 0,
                  f"expect_column_values_to_not_be_null: {col}")

    # === BUSINESS LOGIC VALIDATION ===
    print("   💼 Validating business logic constraints...")

    if "gender" in df.columns:
        check(df["gender"].isin(["Male", "Female"]).all(),
              "expect_column_values_to_be_in_set: gender")

    for col in ["Partner", "Dependents", "PhoneService"]:
        if col in df.columns:
            check(df[col].isin(["Yes", "No"]).all(),
                  f"expect_column_values_to_be_in_set: {col}")

    if "Contract" in df.columns:
        check(df["Contract"].isin(["Month-to-month", "One year", "Two year"]).all(),
              "expect_column_values_to_be_in_set: Contract")

    if "InternetService" in df.columns:
        check(df["InternetService"].isin(["DSL", "Fiber optic", "No"]).all(),
              "expect_column_values_to_be_in_set: InternetService")

    # === NUMERIC RANGE VALIDATION ===
    print("   📊 Validating numeric ranges and business constraints...")

    if "tenure" in df.columns:
        check((df["tenure"].dropna() >= 0).all(),
              "expect_column_values_to_be_between: tenure >= 0")
        check((df["tenure"].dropna() <= 120).all(),
              "expect_column_values_to_be_between: tenure <= 120")

    if "MonthlyCharges" in df.columns:
        check((df["MonthlyCharges"].dropna() >= 0).all(),
              "expect_column_values_to_be_between: MonthlyCharges >= 0")
        check((df["MonthlyCharges"].dropna() <= 200).all(),
              "expect_column_values_to_be_between: MonthlyCharges <= 200")

    if "TotalCharges" in df.columns:
        check((df["TotalCharges"].dropna() >= 0).all(),
              "expect_column_values_to_be_between: TotalCharges >= 0")

    # === DATA CONSISTENCY CHECKS ===
    print("   🔗 Validating data consistency...")

    if "TotalCharges" in df.columns and "MonthlyCharges" in df.columns:
        valid_rows = df[["TotalCharges", "MonthlyCharges"]].dropna()
        mostly_valid = (valid_rows["TotalCharges"] >= valid_rows["MonthlyCharges"]).mean() >= 0.95
        check(mostly_valid,
              "expect_column_pair_values: TotalCharges >= MonthlyCharges (95%)")

    # === STATISTICAL VALIDATION ===
    print("   📈 Validating statistical properties...")

    if "customerID" in df.columns:
        check(df["customerID"].duplicated().sum() == 0,
              "expect_column_values_to_be_unique: customerID")

    check(len(df) > 0, "expect_table_row_count_to_be_greater_than: 0")
    check(len(df) >= 100, "expect_table_row_count_to_be_greater_than: 100")

    # === RESULTS ===
    passed_checks = total_checks - len(failed_expectations)
    is_valid = len(failed_expectations) == 0

    print(f"\n{'='*50}")
    if is_valid:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {len(failed_expectations)}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")
    print(f"{'='*50}\n")

    return is_valid, failed_expectations
