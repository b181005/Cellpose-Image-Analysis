import matplotlib.pyplot as plt
import seaborn as sns



def plot_intensity_correlation(df, x_col='mean_gfp', y_col='mean_s647', hue_col='Condition'):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, s=10, alpha=0.7)
    plt.xlabel("GFP Intensity (Channel 2)")
    plt.ylabel("CH4 Intensity")
    plt.title(f"Correlation of {x_col} vs. {y_col} per Cell")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_condition_violin(df, y_col='ratio_s647_gfp', x_col='Condition'):
    plt.figure(figsize=(8, 6))
    sns.violinplot(
        data=df, x=x_col, y=y_col, hue=x_col, 
        legend=False, palette=['blue', 'gray'], order=['S-B+', 'S+B+']
    )
    plt.title(f"Distribution of {y_col} by {x_col}")
    plt.show()