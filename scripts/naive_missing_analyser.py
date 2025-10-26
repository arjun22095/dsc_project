import pandas as pd
import argparse
import os


def missing_value_report(csv_path, save_path=None):
    # Load CSV file
    df = pd.read_csv(csv_path, low_memory=False)

    # Calculate missing values
    total_rows = len(df)
    missing_counts = df.isnull().sum() + (
        df.eq("").sum()
    )  # counts both NaN and empty strings
    missing_percent = (missing_counts / total_rows) * 100

    # Identify data types
    data_types = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            data_types.append("Numeric")
        else:
            data_types.append("Categorical")

    # Create summary DataFrame
    report = pd.DataFrame(
        {
            "Column Name": df.columns,
            "Type": data_types,
            "Missing Values": missing_counts.values,
            "Total Rows": total_rows,
            "Missing %": missing_percent.round(2).values,
        }
    )

    # Sort by Missing % descending
    report = report.sort_values(by="Missing %", ascending=False).reset_index(drop=True)
    report.index = report.index + 1  # Add a 1-based index

    # Disable truncation for full output
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    # Print report to console
    print(f"\nMissing Value Report for: {csv_path}\n")
    print(report.to_string(index=True))

    # Optionally save to CSV
    if save_path:
        report.to_csv(save_path, index_label="Index")
        print(f"\nReport saved to: {os.path.abspath(save_path)}")


def main():
    parser = argparse.ArgumentParser(
        description="Check each column in a CSV for percentage of missing (empty) values and data types."
    )
    parser.add_argument("csv_path", type=str, help="Path to the CSV file.")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the missing value report as a CSV file.",
    )
    args = parser.parse_args()

    missing_value_report(args.csv_path, args.save)


if __name__ == "__main__":
    main()
