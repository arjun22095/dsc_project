"""
Outputs:
- tables/: missingness.csv, numeric_summary.csv, categorical_summary.csv,
           correlations.csv, group_approval_rates.csv
- figures/: multiple PNGs for categorical counts (bar/pie), histograms/KDE,
           boxplots, scatter plots, stacked bars, and correlation heatmap.

Usage:
  python hmda_eda.py --input ../data/hmda_ne_all_2017.csv --outdir ../reports/hmda_eda --sample_n 500000
"""

import argparse
import os
import warnings
import traceback
from typing import Optional, List, Dict, Any
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk")


# ----------------------------- Plotter Class ----------------------------- #


class HmdaPlotter:
    """Handles all plotting tasks for the HMDA EDA."""

    def __init__(self, df: pd.DataFrame, figs_dir: str) -> None:
        """
        Initialize the plotter.

        Args:
            df: The DataFrame containing data to plot.
            figs_dir: The directory where figure files will be saved.
        """
        self.df: pd.DataFrame = df
        self.figs_dir: str = figs_dir
        self._ensure_dir(self.figs_dir)

    @staticmethod
    def _ensure_dir(path: str) -> None:
        """Ensures a directory exists."""
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _safe_savefig(path: str, bbox_inches: str = "tight", dpi: int = 150) -> None:
        """Saves a plot to a file and closes the figure."""
        plt.savefig(path, bbox_inches=bbox_inches, dpi=dpi)
        plt.close()

    def plot_bar(
        self,
        col: str,
        out_path: str,
        top_n: Optional[int] = None,
        title: Optional[str] = None,
        rotate: int = 0,
        exclude_na: bool = False,
    ) -> None:
        """Creates and saves a bar plot."""
        vc: pd.Series = self.df[col].value_counts(dropna=exclude_na)
        if top_n:
            vc = vc.head(top_n)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=vc.index.astype(str), y=vc.values, palette="deep")
        plt.title(title or f"Distribution: {col}")
        plt.ylabel("Count")
        plt.xlabel(col)
        plt.xticks(rotation=rotate, ha="right")
        for i, v in enumerate(vc.values):
            plt.text(i, v, f"{v}", ha="center", va="bottom", fontsize=10)
        self._safe_savefig(out_path)

    def plot_pie(
        self,
        col: str,
        out_path: str,
        top_n: Optional[int] = None,
        title: Optional[str] = None,
        exclude_na: bool = False,
    ) -> None:
        """Creates and saves a pie chart."""
        vc: pd.Series = self.df[col].value_counts(dropna=exclude_na)
        if top_n:
            vc = vc.head(top_n)
        plt.figure(figsize=(8, 8))
        plt.pie(
            vc.values,
            labels=vc.index.astype(str),
            autopct="%1.1f%%",
            startangle=140,
            textprops={"fontsize": 10},
        )
        plt.title(title or f"Share: {col}")
        self._safe_savefig(out_path)

    def plot_hist_kde(
        self,
        col: str,
        out_path: str,
        bins: int = 50,
        logx: bool = False,
        title: Optional[str] = None,
    ) -> None:
        """Creates and saves a histogram with a KDE overlay."""
        series: pd.Series = pd.to_numeric(self.df[col], errors="coerce").dropna()
        if series.empty:
            print(f"Warning: No data to plot for hist_kde on column '{col}'. Skipping.")
            return
        plt.figure(figsize=(10, 6))
        sns.histplot(series, bins=bins, kde=True, color="#70c1fa")
        if logx:
            plt.xscale("log")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(title or f"Distribution: {col}")
        self._safe_savefig(out_path)

    def plot_box_by_cat(
        self, num_col: str, cat_col: str, out_path: str, title: Optional[str] = None
    ) -> None:
        """Creates and saves a boxplot grouped by a categorical variable."""
        sub: pd.DataFrame = self.df[[num_col, cat_col]].copy()
        sub[num_col] = pd.to_numeric(sub[num_col], errors="coerce")
        sub = sub.dropna()
        if sub.empty:
            print(
                f"Warning: No data to plot for box_by_cat on '{num_col}' by '{cat_col}'. Skipping."
            )
            return
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=sub, x=cat_col, y=num_col)
        sns.stripplot(data=sub, x=cat_col, y=num_col, color="k", size=3, alpha=0.4)
        plt.title(title or f"{num_col} by {cat_col}")
        plt.xticks(rotation=20, ha="right")
        self._safe_savefig(out_path)

    def plot_scatter(
        self,
        x: str,
        y: str,
        hue: Optional[str],
        out_path: str,
        add_trend: bool = True,
        title: Optional[str] = None,
    ) -> None:
        """Creates and saves a scatter plot, optionally with a regression line."""
        cols_to_check: List[str] = [x, y] + ([hue] if hue else [])
        sub: pd.DataFrame = self.df[cols_to_check].copy()
        sub[x] = pd.to_numeric(sub[x], errors="coerce")
        sub[y] = pd.to_numeric(sub[y], errors="coerce")
        sub = sub.dropna(subset=[x, y])
        if sub.empty:
            print(f"Warning: No data to plot for scatter on '{y}' vs '{x}'. Skipping.")
            return

        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=sub, x=x, y=y, hue=hue, alpha=0.6)
        if add_trend:
            sns.regplot(data=sub, x=x, y=y, scatter=False, color="red", ci=None)
        plt.title(title or f"{y} vs {x}")
        plt.tight_layout()
        self._safe_savefig(out_path)

    def plot_stacked_bar(
        self,
        row_cat: str,
        col_cat: str,
        out_path: str,
        normalize: bool = True,
        title: Optional[str] = None,
        exclude_na: bool = False,
    ) -> None:
        """Creates and saves a stacked bar chart."""
        ct: pd.DataFrame = pd.crosstab(
            self.df[row_cat], self.df[col_cat], dropna=exclude_na
        )
        if normalize:
            ct = (ct.T / ct.T.sum()).T
        ct.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="Set2")
        plt.title(title or f"{row_cat} by {col_cat}")
        plt.ylabel("Proportion" if normalize else "Count")
        plt.xticks(rotation=20, ha="right")
        plt.legend(title=col_cat, bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.tight_layout()
        self._safe_savefig(out_path)

    def plot_corr_heatmap(
        self, num_cols: List[str], out_path: str, title: str = "Correlation Heatmap"
    ) -> Optional[pd.DataFrame]:
        """Creates a correlation heatmap and returns the correlation matrix."""
        sub: pd.DataFrame = self.df[num_cols].apply(pd.to_numeric, errors="coerce")
        sub = sub.dropna(how="all", axis=0).dropna(how="all", axis=1)
        if sub.shape[1] < 2:
            print("Warning: Skipping correlation heatmap, < 2 numeric columns found.")
            return None
        corr: pd.DataFrame = sub.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=False)
        plt.title(title)
        plt.tight_layout()
        self._safe_savefig(out_path)
        return corr


# ----------------------------- EDA Runner Class ----------------------------- #


class HmdaEdaRunner:
    """
    Encapsulates the entire HMDA Exploratory Data Analysis pipeline.

    This class handles loading, cleaning, and summarizing data,
    and delegates all plotting tasks to an HmdaPlotter instance.
    """

    COLUMN_CONFIG: Dict[str, Dict[str, Any]] = {
        # --- Numeric Columns ---
        "applicant_income_000s": {
            "type": "numeric",
            "title": "Applicant Income (thousands)",
            "plot_kind": "hist_kde",
        },
        "as_of_year": {
            "type": "numeric",
            "title": "Application Year",
            "plot_kind": "hist_kde",
        },
        "census_tract_number": {"type": "numeric", "plot_kind": None},
        "hud_median_family_income": {
            "type": "numeric",
            "title": "HUD Median Family Income (USD)",
            "plot_kind": "hist_kde",
        },
        "loan_amount_000s": {
            "type": "numeric",
            "title": "Loan Amount (thousands)",
            "plot_kind": "hist_kde",
        },
        "minority_population": {
            "type": "numeric",
            "title": "Minority Population (%)",
            "plot_kind": "hist_kde",
        },
        "number_of_1_to_4_family_units": {
            "type": "numeric",
            "title": "Dwellings (1-4 families)",
            "plot_kind": "hist_kde",
        },
        "number_of_owner_occupied_units": {
            "type": "numeric",
            "title": "Owner-Occupied Dwellings",
            "plot_kind": "hist_kde",
        },
        "population": {
            "type": "numeric",
            "title": "Total population in tract",
            "plot_kind": "hist_kde",
        },
        "tract_to_msamd_income": {
            "type": "numeric",
            "title": "Tract to MSA/MD Income (%)",
            "plot_kind": "hist_kde",
        },
        "rate_spread": {
            "type": "numeric",
            "title": "Rate Spread",
            "plot_kind": "hist_kde",
        },
        # --- Derived Numeric ---
        "loan_to_income_ratio": {
            "type": "numeric",
            "title": "Loan-to-Income Ratio",
            "plot_kind": "hist_kde",
        },
        # --- Categorical Columns ---
        "denial_reason_name_1": {
            "type": "categorical",
            "title": "Top Denial Reasons",
            "plot_kind": "bar",
            "top_n": 10,
            "rotate": 30,
            "exclude_na": True,
        },
        "msamd_name": {
            "type": "categorical",
            "title": "Metropolitan Statistical Area/Metropolitan Division",
            "plot_kind": "pie",
        },
        "county_name": {
            "type": "categorical",
            "title": "Name of the County",
            "plot_kind": "pie",
        },
        "loan_type_name": {
            "type": "categorical",
            "title": "Type of Loan",
            "plot_kind": "pie",
        },
        "agency_name": {"type": "categorical", "plot_kind": None},
        "respondent_id": {"type": "categorical", "plot_kind": None},
        "agency_abbr": {
            "type": "categorical",
            "title": "Distribution by Supervisory Agency",
            "plot_kind": "bar",
            "top_n": 20,
            "rotate": 20,
        },
        "owner_occupancy_name": {
            "type": "categorical",
            "title": "Owner-Occupancy Status",
            "plot_kind": "bar",
            "rotate": 10,
        },
        "loan_purpose_name": {
            "type": "categorical",
            "title": "Distribution of Loan Purposes",
            "plot_kind": "bar",
            "top_n": 20,
            "rotate": 15,
        },
        "property_type_name": {
            "type": "categorical",
            "title": "Type of Property",
            "plot_kind": "pie",
        },
        "preapproval_name": {
            "type": "categorical",
            "title": "PreApproval Request Status",
            "plot_kind": "pie",
        },
        "state_abbr": {"type": "categorical", "plot_kind": None},
        "state_name": {
            "type": "categorical",
            "title": "Name of the State",
            "plot_kind": "bar",
            "top_n": 20,
            "rotate": 20,
        },
        "action_taken_name": {
            "type": "categorical",
            "title": "Application Outcomes",
            "plot_kind": "pie",
        },
        "applicant_race_name_1": {
            "type": "categorical",
            "title": "Applicant Race (Primary)",
            "plot_kind": "bar",
            "top_n": 15,
            "rotate": 20,
        },
        "applicant_ethnicity_name": {
            "type": "categorical",
            "title": "Applicant Ethnicity",
            "plot_kind": "pie",
        },
        "applicant_sex_name": {
            "type": "categorical",
            "title": "Applicant Sex",
            "plot_kind": "pie",
        },
        "purchaser_type_name": {
            "type": "categorical",
            "title": "Purchaser Types",
            "plot_kind": "bar",
            "top_n": 15,
            "rotate": 20,
        },
        "co_applicant_ethnicity_name": {
            "type": "categorical",
            "title": "Co-Applicant's Ethnicity",
            "plot_kind": "pie",
            "exclude_na": True,
        },
        "co_applicant_race_name_1": {
            "type": "categorical",
            "title": "Co-Applicant's Race",
            "plot_kind": "pie",
            "exclude_na": True,
        },
        "co_applicant_sex_name": {
            "type": "categorical",
            "title": "Co-Applicant's Sex",
            "plot_kind": "pie",
            "exclude_na": True,
        },
        "lien_status_name": {
            "type": "categorical",
            "title": "Lien Status",
            "plot_kind": "bar",
            "rotate": 10,
        },
        "hoepa_status_name": {
            "type": "categorical",
            "title": "HOEPA Status",
            "plot_kind": "pie",
        },
        # --- Derived Categorical ---
        "urban_rural": {
            "type": "categorical",
            "title": "Urban vs Rural (MSA Presence)",
            "plot_kind": "pie",
        },
        "minority_category": {
            "type": "categorical",
            "title": "Minority Population Category",
            "plot_kind": "bar",
            "rotate": 10,
        },
        "has_co_applicant": {
            "type": "categorical",
            "title": "Has Co-Applicant",
            "plot_kind": "pie",
        },
    }

    def __init__(
        self,
        input_path: str,
        outdir: str,
        sample_n: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        self.input_path: str = input_path
        self.outdir: str = outdir
        self.sample_n: Optional[int] = sample_n
        self.random_state: int = random_state

        self.tbls_dir: str = os.path.join(self.outdir, "tables")
        self.figs_dir: str = os.path.join(self.outdir, "figures")
        self._ensure_dir(self.tbls_dir)

        self.df: Optional[pd.DataFrame] = None
        self.plotter: Optional[HmdaPlotter] = None

        self.key_numeric: List[str] = []
        self.key_cats: List[str] = []
        self.corr_cols: List[str] = []

        print(f"EDA Runner initialized. Output will be saved to: {self.outdir}")

    # ----------------------------- Utils ----------------------------- #

    @staticmethod
    def _ensure_dir(path: str) -> None:
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _to_snake(s: str) -> str:
        return (
            s.strip()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace("__", "_")
            .lower()
        )

    # ------------------------- Data Load & Prep ------------------------ #

    def _load_hmda(self) -> pd.DataFrame:
        """Loads the HMDA data from the specified input path."""
        print(f"Loading data from {self.input_path}...")
        df: pd.DataFrame = pd.read_csv(
            self.input_path, low_memory=False, encoding="utf-8"
        )
        if self.sample_n is not None and len(df) > self.sample_n:
            print(f"Sampling {self.sample_n} rows...")
            df = df.sample(n=self.sample_n, random_state=self.random_state)
        print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
        return df

    def _clean_columns(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("self.df is not loaded.")
        print("Cleaning columns...")
        df: pd.DataFrame = self.df.rename(
            columns={c: self._to_snake(c) for c in self.df.columns}
        )
        for col in df.select_dtypes(include=["object", "string"]).columns:
            df[col] = df[col].astype("string").str.strip()
        return df

    def _add_derived_features(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("self.df is not loaded.")
        print("Adding derived features...")
        df: pd.DataFrame = self.df.copy()

        if "loan_amount_000s" in df.columns and "applicant_income_000s" in df.columns:
            df["loan_to_income_ratio"] = pd.to_numeric(
                df["loan_amount_000s"], errors="coerce"
            ) / pd.to_numeric(df["applicant_income_000s"], errors="coerce")

        co_sex_name: str = (
            "co_applicant_sex_name" if "co_applicant_sex_name" in df.columns else ""
        )
        if co_sex_name:
            df["has_co_applicant"] = np.where(
                df[co_sex_name].fillna("").str.contains("No co-applicant", case=False),
                "No",
                "Yes",
            )

        msa_cols: List[str] = [c for c in ["msamd_name", "msamd"] if c in df.columns]
        if msa_cols:
            df["urban_rural"] = np.where(
                df[msa_cols].apply(lambda r: r.isna().all(), axis=1),
                "Rural/No MSA",
                "MSA/Metro",
            )

        if "minority_population" in df.columns:
            df["minority_category"] = pd.cut(
                pd.to_numeric(df["minority_population"], errors="coerce"),
                bins=[-np.inf, 25, 50, 75, np.inf],
                labels=[
                    "Low (<25%)",
                    "Medium (25-50%)",
                    "High (50-75%)",
                    "Very High (>75%)",
                ],
            )
        return df

    def _identify_columns(self) -> None:
        """Identifies and stores key column lists from the central config."""
        if self.df is None:
            raise ValueError("self.df is not loaded.")

        print("Identifying columns from config...")

        # Populate lists based on config AND presence in the DataFrame
        for col_name, config in self.COLUMN_CONFIG.items():
            if col_name in self.df.columns:
                col_type = config.get("type")
                if col_type == "numeric":
                    self.key_numeric.append(col_name)
                elif col_type == "categorical":
                    self.key_cats.append(col_name)

        # Correlation columns are all numeric columns found
        # (Could also add a "use_for_corr" flag in config for more control)
        self.corr_cols = self.key_numeric

        print(
            f"Found {len(self.key_numeric)} numeric, {len(self.key_cats)} categorical columns for analysis."
        )

    def load_and_prepare_data(self) -> None:
        self.df = self._load_hmda()
        self.df = self._clean_columns()
        self.df = self._add_derived_features()
        self._identify_columns()

        self.plotter = HmdaPlotter(self.df, self.figs_dir)
        print("Plotter initialized.")

    # ----------------------------- Summaries ----------------------------- #

    def generate_summaries(self) -> None:
        if self.df is None:
            raise ValueError("Data not loaded. Run load_and_prepare_data() first.")

        print("Generating summary tables...")
        # Missingness
        miss: pd.Series = self.df.isna().sum()
        missing_summary: pd.DataFrame = pd.DataFrame(
            {
                "column": miss.index,
                "missing_count": miss.values,
                "missing_pct": (miss.values / len(self.df) * 100).round(2),
            }
        ).sort_values(["missing_pct", "missing_count"], ascending=False)
        missing_summary.to_csv(
            os.path.join(self.tbls_dir, "missingness.csv"), index=False
        )

        # Numeric Summary
        if self.key_numeric:
            desc: pd.DataFrame = (
                self.df[self.key_numeric]
                .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
                .T
            )
            desc = desc.rename(
                columns={
                    "50%": "median",
                    "25%": "p25",
                    "75%": "p75",
                    "5%": "p05",
                    "95%": "p95",
                },
                errors="ignore",
            )
            desc.to_csv(os.path.join(self.tbls_dir, "numeric_summary.csv"))

        # Categorical Summary
        if self.key_cats:
            rows: List[Dict[str, Any]] = []
            for c in self.key_cats:
                vc: pd.Series = self.df[c].value_counts(dropna=False)
                top: pd.Series = vc.head(20)
                for idx, val in top.items():
                    rows.append(
                        {
                            "feature": c,
                            "category": str(idx),
                            "count": int(val),
                            "pct": round(val / len(self.df) * 100, 2),
                        }
                    )
            cat_summary: pd.DataFrame = pd.DataFrame(rows)
            cat_summary.to_csv(
                os.path.join(self.tbls_dir, "categorical_summary.csv"), index=False
            )

    # ------------------- Plotting (Delegated) -------------------- #

    def generate_plots(self) -> None:
        """Generates and saves all visualizations by delegating to the plotter."""
        if self.plotter is None or self.df is None:
            raise ValueError(
                "Plotter or DataFrame not initialized. Run load_and_prepare_data() first."
            )

        print("Generating plots (delegating to plotter)...")

        # --- 1. Univariate Plots (from COLUMN_CONFIG) ---
        print("Generating univariate plots...")
        for col_name, config in self.COLUMN_CONFIG.items():
            if col_name not in self.df.columns:
                continue  # Skip columns not in our (sampled) dataframe

            plot_kind = config.get("plot_kind")
            if not plot_kind:
                continue  # Skip columns with "plot_kind": None

            title = config.get("title", f"Distribution: {col_name}")
            out_path = os.path.join(self.figs_dir, f"{col_name}_{plot_kind}.png")

            if plot_kind == "bar":
                self.plotter.plot_bar(
                    col_name,
                    out_path,
                    top_n=config.get("top_n"),
                    title=title,
                    rotate=config.get("rotate", 0),
                    exclude_na=config.get("exclude_na", False),
                )
            elif plot_kind == "pie":
                self.plotter.plot_pie(
                    col_name,
                    out_path,
                    top_n=config.get("top_n"),
                    title=title,
                    exclude_na=config.get("exclude_na", False),
                )
            elif plot_kind == "hist_kde":
                self.plotter.plot_hist_kde(
                    col_name,
                    out_path,
                    bins=config.get("bins", 50),
                    logx=config.get("logx", False),
                    title=title,
                )

        # --- 2. Bivariate/Complex Plots ---
        print("Generating boxplots...")
        if "loan_purpose_name" in self.df.columns:
            for num_col in [
                "loan_amount_000s",
                "applicant_income_000s",
                "loan_to_income_ratio",
                "rate_spread",
            ]:
                if num_col in self.df.columns:
                    title_prefix = self.COLUMN_CONFIG.get(num_col, {}).get(
                        "title", num_col
                    )
                    self.plotter.plot_box_by_cat(
                        num_col,
                        "loan_purpose_name",
                        os.path.join(self.figs_dir, f"box_{num_col}_by_purpose.png"),
                        f"{title_prefix} by Loan Purpose",
                    )

        print("Generating scatter plots...")
        if all(
            c in self.df.columns for c in ["applicant_income_000s", "loan_amount_000s"]
        ):
            hue: Optional[str] = (
                "loan_purpose_name" if "loan_purpose_name" in self.df.columns else None
            )
            self.plotter.plot_scatter(
                "applicant_income_000s",
                "loan_amount_000s",
                hue,
                os.path.join(self.figs_dir, "scatter_income_vs_loan.png"),
                add_trend=True,
                title="Loan Amount vs Applicant Income",
            )

        if all(c in self.df.columns for c in ["applicant_income_000s", "rate_spread"]):
            hue: Optional[str] = (
                "action_taken_name" if "action_taken_name" in self.df.columns else None
            )
            self.plotter.plot_scatter(
                "applicant_income_000s",
                "rate_spread",
                hue,
                os.path.join(self.figs_dir, "scatter_income_vs_rate_spread.png"),
                add_trend=False,
                title="Rate Spread vs Applicant Income",
            )

        print("Generating stacked bars...")
        if "action_taken_name" in self.df.columns:
            for demo_col in [
                "applicant_ethnicity_name",
                "applicant_race_name_1",
                "applicant_sex_name",
            ]:
                if demo_col in self.df.columns:
                    self.plotter.plot_stacked_bar(
                        demo_col,
                        "action_taken_name",
                        os.path.join(
                            self.figs_dir, f"stacked_{demo_col}_by_action.png"
                        ),
                        normalize=True,
                        title=f"{demo_col} by Action Taken (Proportion)",
                    )

    def generate_correlations(self) -> None:
        if self.plotter is None:
            raise ValueError(
                "Plotter not initialized. Run load_and_prepare_data() first."
            )
        print("Generating correlation matrix...")
        if self.corr_cols:
            corr_mat: Optional[pd.DataFrame] = self.plotter.plot_corr_heatmap(
                self.corr_cols,
                os.path.join(self.figs_dir, "correlation_heatmap.png"),
                title="Correlation Matrix of Key HMDA Features",
            )
            if corr_mat is not None:
                corr_mat.to_csv(os.path.join(self.tbls_dir, "correlations.csv"))

    def generate_approval_rates(self) -> None:
        """Calculates and saves grouped approval rates."""
        if self.df is None:
            raise ValueError("Data not loaded. Run load_and_prepare_data() first.")

        print("Generating group approval rates...")
        if (
            "action_taken_name" in self.df.columns
            and "applicant_ethnicity_name" in self.df.columns
        ):
            grp: pd.DataFrame = (
                self.df.groupby("applicant_ethnicity_name")["action_taken_name"]
                .value_counts(normalize=True)
                .rename("proportion")
                .reset_index()
            )
            grp.to_csv(
                os.path.join(self.tbls_dir, "group_approval_rates.csv"), index=False
            )

    def _create_manifest(self) -> None:
        """Saves a JSON manifest of the EDA run."""
        if self.df is None:
            raise ValueError("Data not loaded. Cannot create manifest.")

        print("Creating manifest file...")
        manifest: Dict[str, Any] = {
            "input_path": self.input_path,
            "n_rows": int(len(self.df)),
            "n_cols": int(self.df.shape[1]),
            "figures_dir": self.figs_dir,
            "tables_dir": self.tbls_dir,
            "numeric_cols_used_for_corr": self.corr_cols,
        }
        with open(os.path.join(self.outdir, "eda_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    def run(self) -> None:
        try:
            self.load_and_prepare_data()
            self.generate_summaries()
            self.generate_plots()
            self.generate_correlations()
            self.generate_approval_rates()
            self._create_manifest()
            print(f"\n[OK] EDA complete. Reports saved under: {self.outdir}")
        except Exception as e:
            print(f"\n[ERROR] EDA failed: {e}")
            traceback.print_exc()


# ----------------------------- CLI ----------------------------- #


def parse_args() -> argparse.Namespace:
    ap: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Comprehensive HMDA EDA with visualization."
    )
    ap.add_argument(
        "--input", required=True, help="Path to HMDA CSV (2007–2017 MLAR or similar)."
    )
    ap.add_argument("--outdir", required=True, help="Output directory for reports.")
    ap.add_argument(
        "--sample_n",
        type=int,
        default=None,
        help="Optional row sample for faster runs.",
    )
    return ap.parse_args()


def main() -> None:
    args: argparse.Namespace = parse_args()

    runner: HmdaEdaRunner = HmdaEdaRunner(
        input_path=args.input, outdir=args.outdir, sample_n=args.sample_n
    )
    runner.run()


if __name__ == "__main__":
    main()
