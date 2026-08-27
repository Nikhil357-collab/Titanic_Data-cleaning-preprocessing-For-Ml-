import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    FIGURES_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    IQR_MULTIPLIER
)
from src.reporter import create_report
from src.loader import load_data, split_features_target
from src.explorer import show_basic_info, get_column_types
from src.cleaner import (
    clean_column_names,
    standardize_missing_values,
    remove_duplicates
)
from src.outlier import calculate_iqr_bounds, remove_outliers_iqr
from src.preprocessor import build_preprocessor, transform_to_dataframe
from src.visualiser import create_boxplots


def main():
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    report_folder = FIGURES_PATH.parent / "reports"
    report_folder.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    df = load_data(RAW_DATA_PATH)
    raw_df_for_report = df.copy()

    # 2. Clean column names and missing-value markers
    df = clean_column_names(df)

    df = standardize_missing_values(df)
    rows_before_deduplication = len(df)
    df = remove_duplicates(df)
    duplicate_rows_removed = rows_before_deduplication - len(df)

    # 3. Initial exploration
    show_basic_info(df)
    print("\nTarget column values:")
    print(df["survived"].value_counts())
    # 4. Separate features and target
    target_column = TARGET_COLUMN.strip().lower().replace(" ", "_")
    X, y = split_features_target(df, target_column)
    # 5. Identify numerical and categorical columns
    numerical_columns, categorical_columns = get_column_types(X)

    print("\nNumerical Columns:")
    print(numerical_columns)

    print("\nCategorical Columns:")
    print(categorical_columns)

    # 6. Split data before fitting preprocessors
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # 7. Visualize training data before outlier removal
    create_boxplots(
        X_train,
        numerical_columns,
        FIGURES_PATH / "boxplot_before_outlier_removal.png",
        "Boxplots Before Outlier Removal"
    )

    # 8. Learn IQR ranges using training data and remove training outliers
    iqr_bounds = calculate_iqr_bounds(
        X_train,
        numerical_columns,
        multiplier=IQR_MULTIPLIER
    )

    X_train_clean = remove_outliers_iqr(X_train, iqr_bounds)
    y_train_clean = y_train.loc[X_train_clean.index]
    outlier_rows_removed = len(X_train) - len(X_train_clean)

    # 9. Visualize training data after outlier removal
    create_boxplots(
        X_train_clean,
        numerical_columns,
        FIGURES_PATH / "boxplot_after_outlier_removal.png",
        "Boxplots After Outlier Removal"
    )

    # 10. Build and run preprocessing pipeline
    preprocessor = build_preprocessor(
        numerical_columns,
        categorical_columns
    )

    X_train_processed, X_test_processed = transform_to_dataframe(
        preprocessor,
        X_train_clean,
        X_test
    )

    # 11. Save final processed datasets
    train_processed = X_train_processed.copy()
    train_processed[target_column] = y_train_clean.values

    test_processed = X_test_processed.copy()
    test_processed[target_column] = y_test.values

    final_dataset = pd.concat([train_processed, test_processed], axis=0)
    final_dataset.to_csv(PROCESSED_DATA_PATH, index=False)
    
    # 12. Generate preprocessing report
    print("\nStarting automated report generation...")

    create_report(
        raw_df=raw_df_for_report,
        final_df=final_dataset,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        duplicate_rows_removed=duplicate_rows_removed,
        outlier_rows_removed= outlier_rows_removed,
        target_column= "survived"
   )

    print("\n--- Final Output ---")
    print(f"Processed training shape: {train_processed.shape}")
    print(f"Processed testing shape: {test_processed.shape}")
    print(f"Final dataset saved to: {PROCESSED_DATA_PATH}")
    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()