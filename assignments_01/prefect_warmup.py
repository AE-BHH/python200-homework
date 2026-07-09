from prefect import task, flow
from prefect.logging import get_run_logger
import pandas as pd
import numpy as np


@task
def create_series(arr):
    return pd.Series(arr, name="Values")


@task
def clean_data(series):
    return series.dropna()


@task
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0],
    }


@flow
def pipeline_flow(arr):
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)
    return summary


if __name__ == "__main__":
    arr = np.array(
        [12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0]
    )

    pipeline_flow(arr)

# 1. This pipeline is simple -- just three small functions on a handful of numbers. Why might Prefect be more overhead than it is worth here?
# Answer: Prefect is designed for more complex workflows that involve multiple tasks but in this case, the pipeline is simple which doesn't require that level of orchestration.

# 2. Describe some realistic scenarios where a framework like Prefect could still be useful, even if the pipeline logic itself stays simple like in this case.
# Answer: Prefect could be useful in scenarios where the tasks need to be scheduled to run at specific times, or when there is a need for monitoring and logging of task execution.
