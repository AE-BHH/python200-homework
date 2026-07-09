# Pandas
# Pandas Q1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns

data = {
    "name": ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade": [85, 72, 90, 68, 95],
    "city": ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True],
}
df = pd.DataFrame(data)

print(f"first three rows: {df.head(3)}")
print(f"dataframe shape: {df.shape}")
print(f"dataframe info: {df.info()}")

# Pandas Q2

filtered_df = df[df["grade"] > 80]
print(filtered_df)

# Pandas Q3

df["grade_curved"] = df["grade"] + 5
print(f"New Dataframe: {df}")


# Pandas Q4

df["name_upper"] = df["name"].str.upper()
print(f"changed name to uppercase: {df[['name', 'name_upper']]}")

# Pandas Q5
df_grouped = df.groupby("city")["grade"].mean()
print(f"mean grade per city: {df_grouped}")

# Pandas Q6
df["city"] = df["city"].replace("Austin", "Houston")
print(df[["name", "city"]])

# Pandas Q7
sorted_df = df.sort_values(by="grade", ascending=False)
print(f"sorted dataframe: {sorted_df}")

# Numpy Q1
arr = np.array([10, 20, 30, 40, 50])
print(f"array shape: {arr.shape}")
print(f"array data type: {arr.dtype}")
print(f"array number of dim: {arr.ndim}")

# Numpy Q2
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"array size: {arr.size}")
print(f"array shape: {arr.shape}")

# Numpy Q3
print(f"sliced array: {arr[0:2, 0:2]}")

# Numpy Q4
print(f"array 3x4: {np.zeros((3, 4))}")
print(f"array 2x5: {np.ones((2, 5))}")

# Numpy Q5
arr = np.arange(0, 50, 5)
print(f"printed array: {arr}")
print(f"array shape: {arr.shape}")
print(f"array mean: {np.mean(arr)}")
print(f"array sum: {np.sum(arr)}")
print(f"standard deviation: {np.std(arr)}")

# Numpy Q6
mu, sigma = 0, 1
s = np.random.normal(mu, sigma, 200)
print(f"mean: {np.mean(s)}")
print(f"standard deviation: {np.std(s)}")

# Matplotlib Q1
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.plot(x, y)
plt.title("Squares")
plt.xlabel(x)
plt.ylabel(y)
plt.show()

# Matplotlib Q2
subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]
plt.bar(subjects, scores)
plt.title("Student Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.show()


# Matplotlib Q3
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.scatter(x1, y1, color="blue", label="Dataset 1")
plt.scatter(x2, y2, color="red", label="Dataset 2")
plt.title("Scatter Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()


# Matplotlib Q4

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(x, y)
axes[0].set_title("Squares")
axes[0].set_xlabel("X-axis")
axes[0].set_ylabel("Y-axis")

axes[1].bar(subjects, scores)
axes[1].set_title("Student Scores")
axes[1].set_xlabel("Subjects")
axes[1].set_ylabel("Scores")

plt.tight_layout()
plt.show()

# Descriptive Stats Q1

data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
mean = np.mean(data)
median = np.median(data)
variance = np.var(data)
std = np.std(data)
print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Variance: {variance}")
print(f"Standard Deviation: {std}")


# Descriptive Stats Q2
normal_data = np.random.normal(loc=65, scale=10, size=500)
plt.hist(normal_data, bins=20, color="blue", edgecolor="black")
plt.title("Distribution of Scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")
plt.show()


# Descriptive Stats Q3
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.ylabel("Scores")
plt.show()

# Descriptive Stats Q4
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)
plt.boxplot([normal_data, skewed_data], labels=["Normal", "Skewed"])
plt.title("Distribution Comarison")
plt.ylabel("Values")
plt.show()

# Exponential is more skewed (right-skewed). Mean is fine for Normal;
# median is better for Exponential since its mean is pulled by the tail.

# Descriptive Stats Q5
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

mean1 = np.mean(data1)
median1 = np.median(data1)
mode1 = stats.mode(data1)

mean2 = np.mean(data2)
median2 = np.median(data2)
mode2 = stats.mode(data2)

print(f"Data 1: Mean: {mean1}, Median: {median1}, Mode: {mode1}")
print(f"Data 2: Mean: {mean2}, Median: {median2}, Mode: {mode2}")

# Why are the median and mean so different for data2?
# The mean is significantly affected by the outlier "150", which skews the ava to the right, while the median is more robust and remains unaffected by the outlier.


# Hypothesis Testing Q1

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_test, p_value = stats.ttest_ind(group_a, group_b)
print("Answer for Hypothesis Testing Q1:")
print(f"t-test statistic:", {t_test})
print(f"p-value statistic:", {p_value})

# Hypothesis Testing Q2
if p_value < 0.05:
    print("The difference is statistically significant.")
else:
    print("The difference is not statistically significant.")

# Hypothesis Testing Q3
before = [60, 65, 70, 58, 62, 67, 63, 66]
after = [68, 70, 76, 65, 69, 72, 70, 71]

t_test, p_value = stats.ttest_rel(before, after)
print("Answer for Hypothesis Testing Q3:")
print(f"t-test statistic: {t_test:.3f}")
print(f"p-value statistic: {p_value:.5f}")

# Hypothesis Testing Q4
scores = [72, 68, 75, 70, 69, 74, 71, 73]
t_test, p_value = stats.ttest_1samp(scores, 70)
print("Answer for Hypothesis Testing Q4:")
print(f"t-test statistic: {t_test:.3f}")
print(f"p-value statistic: {p_value:.5f}")


# Hypothesis Testing Q5
stats.ttest_ind(group_a, group_b, alternative="less")
print("Answer for Hypothesis Testing Q5:")
print(f"t-test statistic: {t_test:.3f}")
print(f"p-value statistic: {p_value:.5f}")

# Hypothesis Testing Q6
print("Answer for Hypothesis Testing Q6:")
print(
    "Group B has higher values than Group A, and this difference is very unlikely to be due to chance (p-value is extremely small). This suggests a real difference between the groups."
)

# Correlation Q1
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

corr_matrix = np.corrcoef(x, y)
print("Answer for Correlation Q1:")
print(corr_matrix)
print(f"Correlation coefficient: {corr_matrix[0, 1]:.3f}")

print(
    "I expect the correlation to be 1 because y is exactly 2 times x (y = 2x), which means there is a perfect positive linear relationship between x and y."
)

# Correlation Q2
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [10, 9, 7, 8, 6, 5, 3, 4, 2, 1]

r, p = pearsonr(x, y)
print("Answer for Correlation Q2:")
print("Correlation:", round(r, 2))
print("p-value:", round(p, 4))

# Correlation Q3
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55, 60, 65, 72, 80],
    "age": [25, 30, 22, 35, 28],
}
df = pd.DataFrame(people)
corr = df.corr()
print("Answer for Correlation Q3:")
print(corr)

# Correlation Q4
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.scatter(x, y)
plt.title("Negative Correlation")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid()
plt.show()

# Correlation Q5
df = pd.DataFrame(people)
corr = df.corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Pipelines Q1
arr = np.array(
    [12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0]
)


def create_series(arr):
    return pd.Series(arr, name="Values")


def clean_data(series):
    return series.dropna()


def summerize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0],
    }


def data_pipeline(arr):
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summerize_data(cleaned_series)
    return summary


print("Answer for Pipelines Q1:")
result = data_pipeline(arr)
for key, value in result.items():
    print(f"{key}: {value}")
