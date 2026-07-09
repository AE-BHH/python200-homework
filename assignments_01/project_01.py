import os
from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from prefect import flow, task, get_run_logger

DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "python200-homework"
    / "assignments"
    / "resources"
    / "happiness_project"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
YEARS = list(range(2015, 2025))


# Task 1: Load and merge all years
@task(retries=3, retry_delay_seconds=2)
def load_data() -> pd.DataFrame:
    logger = get_run_logger()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    COLUMN_MAP = {
        "Country": "country",
        "Regional indicator": "region",
        "Happiness score": "happiness_score",
        "Ladder score": "happiness_score",
        "GDP per capita": "gdp_per_capita",
        "Social support": "social_support",
        "Healthy life expectancy": "health",
        "Freedom to make life choices": "freedom",
        "Generosity": "generosity",
        "Perceptions of corruption": "corruption",
    }

    frames = []
    for year in YEARS:
        path = os.path.join(DATA_DIR, f"world_happiness_{year}.csv")
        df_year = pd.read_csv(path, sep=";", decimal=",")
        df_year = df_year.rename(columns=COLUMN_MAP)
        df_year = df_year.drop(columns=["Ranking"], errors="ignore")
        df_year["year"] = year
        frames.append(df_year)
        logger.info(f"Loaded {len(df_year)} rows from {path}")

    merged = pd.concat(frames, ignore_index=True)

    out_path = os.path.join(OUTPUT_DIR, "merged_happiness.csv")
    merged.to_csv(out_path, index=False)
    logger.info(f"Merged dataset saved to {out_path} ({len(merged)} rows total)")
    return merged


# Task 2: Descriptive statistics
@task
def descriptive_stats(df: pd.DataFrame) -> dict:
    logger = get_run_logger()

    overall_mean = df["happiness_score"].mean()
    overall_median = df["happiness_score"].median()
    overall_std = df["happiness_score"].std()
    logger.info(
        f"Overall happiness_score -- mean: {overall_mean:.3f}, "
        f"median: {overall_median:.3f}, std: {overall_std:.3f}"
    )

    by_year = df.groupby("year")["happiness_score"].mean().sort_index()
    logger.info(f"Mean happiness by year:\n{by_year.to_string()}")

    by_region = (
        df.groupby("region")["happiness_score"].mean().sort_values(ascending=False)
    )
    logger.info(f"Mean happiness by region:\n{by_region.to_string()}")

    return {
        "overall_mean": overall_mean,
        "overall_median": overall_median,
        "overall_std": overall_std,
        "by_year": by_year,
        "by_region": by_region,
    }


# Task 3: Visual exploration
@task
def visual_exploration(df: pd.DataFrame) -> None:
    logger = get_run_logger()
    numeric_cols = df.select_dtypes(include="number").drop(
        columns=["year"], errors="ignore"
    )

    # Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(df["happiness_score"], bins=25, kde=True)
    plt.title("Distribution of Happiness Scores (All Years)")
    plt.xlabel("Happiness Score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "happiness_histogram.png"))
    plt.close()
    logger.info("Saved happiness_histogram.png")

    # Boxplot by year
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df.assign(year=df["year"].astype(str)), x="year", y="happiness_score"
    )
    plt.title("Happiness Score Distribution by Year")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "happiness_by_year.png"))
    plt.close()
    logger.info("Saved happiness_by_year.png")

    # Scatter: GDP vs happiness
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="gdp_per_capita",
        y="happiness_score",
        hue="region",
        legend=False,
        alpha=0.6,
    )
    plt.title("GDP per Capita vs Happiness Score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gdp_vs_happiness.png"))
    plt.close()
    logger.info("Saved gdp_vs_happiness.png")

    # Correlation heatmap
    plt.figure(figsize=(9, 7))
    corr = numeric_cols.corr(method="pearson")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Pearson Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
    plt.close()
    logger.info("Saved correlation_heatmap.png")


# Task 4: Hypothesis testing
@task
def hypothesis_testing(df: pd.DataFrame, by_region: pd.Series) -> dict:
    logger = get_run_logger()
    alpha = 0.05

    # Test 1: pre- vs post-2020 (pandemic)
    pre = df[df["year"] == 2019]["happiness_score"]
    post = df[df["year"] == 2020]["happiness_score"]
    t_stat, p_val = stats.ttest_ind(pre, post, equal_var=False)

    mean_pre, mean_post = pre.mean(), post.mean()
    logger.info(f"[Pandemic t-test] t={t_stat:.4f}, p={p_val:.4f}")
    logger.info(
        f"[Pandemic t-test] 2019 mean: {mean_pre:.3f}, 2020 mean: {mean_post:.3f}"
    )

    if p_val < alpha:
        direction = "lower" if mean_post < mean_pre else "higher"
        interp1 = (
            f"The difference in happiness scores between 2019 and 2020 is statistically "
            f"significant (p={p_val:.4f} < {alpha}). Average happiness in 2020 was {direction} "
            f"than in 2019 by {abs(mean_pre - mean_post):.3f} points, which is consistent with "
            f"the onset of the COVID-19 pandemic having a measurable association with reported "
            f"global happiness. Note this is an association from observational data, not "
            f"causal proof."
        )
    else:
        interp1 = (
            f"The difference in happiness scores between 2019 and 2020 is not statistically "
            f"significant (p={p_val:.4f} >= {alpha}). We cannot conclude that average global "
            f"happiness changed meaningfully in the first year of the pandemic, at least as "
            f"measured by this survey-based score."
        )
    logger.info(f"[Pandemic t-test] Interpretation: {interp1}")

    # Test 2: two regions expected to differ, chosen from the earlier descriptive stats
    top_region = by_region.index[0]
    bottom_region = by_region.index[-1]
    group_top = df[df["region"] == top_region]["happiness_score"]
    group_bottom = df[df["region"] == bottom_region]["happiness_score"]
    t_stat2, p_val2 = stats.ttest_ind(group_top, group_bottom, equal_var=False)

    logger.info(
        f"[Region t-test] Comparing '{top_region}' (highest mean) vs "
        f"'{bottom_region}' (lowest mean): t={t_stat2:.4f}, p={p_val2:.4f}"
    )
    if p_val2 < alpha:
        interp2 = (
            f"'{top_region}' and '{bottom_region}' have significantly different mean "
            f"happiness scores (p={p_val2:.4f}), consistent with the large gap seen in the "
            f"descriptive statistics."
        )
    else:
        interp2 = (
            f"Despite differing means in the descriptive statistics, the difference between "
            f"'{top_region}' and '{bottom_region}' is not statistically significant "
            f"(p={p_val2:.4f}), which may be due to high within-region variance or a small "
            f"sample of countries."
        )
    logger.info(f"[Region t-test] Interpretation: {interp2}")

    return {
        "pandemic_t": t_stat,
        "pandemic_p": p_val,
        "pandemic_mean_pre": mean_pre,
        "pandemic_mean_post": mean_post,
        "pandemic_interpretation": interp1,
        "region_t": t_stat2,
        "region_p": p_val2,
        "region_top": top_region,
        "region_bottom": bottom_region,
        "region_interpretation": interp2,
    }


# Task 5: Correlation + multiple comparisons correction
@task
def correlation_analysis(df: pd.DataFrame) -> dict:
    logger = get_run_logger()
    alpha = 0.05

    explanatory_vars = [
        c
        for c in df.select_dtypes(include="number").columns
        if c not in ("happiness_score", "year")
    ]

    results = {}
    for col in explanatory_vars:
        valid = df[[col, "happiness_score"]].dropna()
        r, p = stats.pearsonr(valid[col], valid["happiness_score"])
        results[col] = {"r": r, "p": p}
        logger.info(f"Correlation happiness_score ~ {col}: r={r:.4f}, p={p:.4g}")

    n_tests = len(explanatory_vars)
    adjusted_alpha = alpha / n_tests
    logger.info(
        f"Number of correlation tests run: {n_tests}. Bonferroni-adjusted alpha: {adjusted_alpha:.5g}"
    )

    sig_uncorrected = [c for c, v in results.items() if v["p"] < alpha]
    sig_corrected = [c for c, v in results.items() if v["p"] < adjusted_alpha]

    logger.info(f"Significant at alpha=0.05 (uncorrected): {sig_uncorrected}")
    logger.info(f"Still significant after Bonferroni correction: {sig_corrected}")

    if len(sig_uncorrected) > len(sig_corrected):
        dropped = set(sig_uncorrected) - set(sig_corrected)
        logger.info(
            f"The following correlation(s) looked significant before correction but did not "
            f"survive the stricter Bonferroni threshold, and should be treated with caution: "
            f"{sorted(dropped)}"
        )

    strongest_corrected = None
    if sig_corrected:
        strongest_corrected = max(sig_corrected, key=lambda c: abs(results[c]["r"]))

    return {
        "results": results,
        "n_tests": n_tests,
        "adjusted_alpha": adjusted_alpha,
        "sig_uncorrected": sig_uncorrected,
        "sig_corrected": sig_corrected,
        "strongest_corrected": strongest_corrected,
    }


# Task 6: Summary report
@task
def summary_report(df: pd.DataFrame, desc: dict, hyp: dict, corr: dict) -> None:
    logger = get_run_logger()

    n_countries = df["country"].nunique()
    n_years = df["year"].nunique()
    logger.info(
        f"SUMMARY: Dataset covers {n_countries} countries across {n_years} years ({YEARS[0]}-{YEARS[-1]})."
    )

    by_region = desc["by_region"]
    top3 = by_region.head(3)
    bottom3 = by_region.tail(3)
    logger.info(
        f"SUMMARY: Top 3 regions by mean happiness: {list(zip(top3.index, top3.round(3)))}"
    )
    logger.info(
        f"SUMMARY: Bottom 3 regions by mean happiness: {list(zip(bottom3.index, bottom3.round(3)))}"
    )

    logger.info(
        f"SUMMARY: Pandemic (2019 vs 2020) finding: {hyp['pandemic_interpretation']}"
    )

    if corr["strongest_corrected"]:
        var = corr["strongest_corrected"]
        r = corr["results"][var]["r"]
        logger.info(
            f"SUMMARY: The variable most strongly correlated with happiness_score after "
            f"Bonferroni correction is '{var}' (r={r:.3f})."
        )
    else:
        logger.info(
            "SUMMARY: No variables remained significantly correlated with happiness_score after Bonferroni correction."
        )


# Flow
@flow(name="happiness_pipeline")
def happiness_pipeline():
    merged = load_data()
    desc = descriptive_stats(merged)
    visual_exploration(merged)
    hyp = hypothesis_testing(merged, desc["by_region"])
    corr = correlation_analysis(merged)
    summary_report(merged, desc, hyp, corr)


if __name__ == "__main__":
    happiness_pipeline()
