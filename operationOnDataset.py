import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# setting up the vibe for visuals


sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100


# Load the data

df = pd.read_csv(r"D:\lpu things\healthcare_treatments__csv.csv")

# fill in missing treatment costs with the median value (seems fair)


if 'treatment_cost' in df.columns:
    df['treatment_cost'] = df['treatment_cost'].fillna(df['treatment_cost'].median())

# get rid of any exact duplicate rows


df.drop_duplicates(inplace=True)

# take a smaller sample if the dataset is huge (saves time)


df_plot = df.copy()
if len(df) > 10000:
    df_plot = df.sample(10000, random_state=42)


# Quick overview

print("Dataset Summary:")
print(df.describe(include='all'))


# Visuals 🎨


# treatment cost distribution (see how it's spread out)


plt.figure(figsize=(8, 5))
sns.kdeplot(data=df_plot, x='treatment_cost', fill=True, color='skyblue', linewidth=2)
plt.title("Distribution of Treatment Costs")
plt.xlabel("Treatment Cost")
plt.ylabel("Density")
plt.tight_layout()
plt.show()

# treatment cost by gender (violin plot gives a nice feel for spread + outliers)


if 'gender' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=df_plot, x='gender', y='treatment_cost')
    plt.title("Treatment Cost by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Treatment Cost")
    plt.tight_layout()
    plt.show()

# swarm plot – gender vs cost, showing individual points (filtered a bit to keep it readable)


if 'gender' in df.columns and 'age' in df.columns:
    df_swarm = df_plot.copy()
    df_swarm = df_swarm[df_swarm['age'].notna()]
    df_swarm = df_swarm[df_swarm['treatment_cost'] < df_swarm['treatment_cost'].quantile(0.95)]

    plt.figure(figsize=(10, 6))
    sns.swarmplot(data=df_swarm, x='gender', y='treatment_cost', hue='gender', palette='coolwarm', size=3)
    plt.title("Gender-wise Treatment Cost (Filtered Swarm View)")
    plt.xlabel("Gender")
    plt.ylabel("Treatment Cost")
    plt.tight_layout()
    plt.show()

# average cost for top 5 most expensive treatment types


if 'treatment_type' in df.columns:
    avg_cost = df.groupby('treatment_type')['treatment_cost'].mean().sort_values(ascending=False).head(5)

    plt.figure(figsize=(9, 5))
    sns.barplot(x=avg_cost.values, y=avg_cost.index)
    plt.title("Top 5 Most Expensive Treatment Types (Avg. Cost)")
    plt.xlabel("Average Cost")
    plt.ylabel("Treatment Type")
    plt.tight_layout()
    plt.show()

# heatmap to check correlations between numeric columns


plt.figure(figsize=(8, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='YlGnBu', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# boxplot – treatment cost by age groups (outliers trimmed for clarity)


if 'age' in df.columns:
    df_plot['age_group'] = pd.cut(df_plot['age'], bins=[0, 18, 35, 50, 65, 100],
                                  labels=["0-18", "19-35", "36-50", "51-65", "65+"])
    df_plot = df_plot[df_plot['treatment_cost'] < df_plot['treatment_cost'].quantile(0.99)]

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='age_group', y='treatment_cost', data=df_plot)
    plt.title("Treatment Cost by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Treatment Cost")
    plt.tight_layout()
    plt.show()
