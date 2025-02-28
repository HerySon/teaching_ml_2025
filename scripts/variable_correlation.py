import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def analyze_numeric_relationships(df, corr_method="pearson", reg_method="linear"):
    """
    Analyze relationships between numerical variables using correlation matrix and regression.

    Parameters:
        df (pd.DataFrame): DataFrame containing numerical variables.
        corr_method (str): Correlation method ('pearson', 'spearman', 'kendall').
        reg_method (str): Regression method ('linear', 'polynomial', 'ridge', 'lasso').

    Returns:
        None
    """
    numeric_df = df.select_dtypes(include=[np.number])

    # Correlation matrix
    plt.figure(figsize=(12, 8))
    corr_matrix = numeric_df.corr(method=corr_method)
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(f"Correlation Matrix ({corr_method} method)")
    plt.show()

    print("Correlation Matrix:")
    print(corr_matrix)

    # Regression between all pairs of numeric columns
    for i, x_col in enumerate(numeric_df.columns):
        for y_col in numeric_df.columns[i+1:]:
            X = numeric_df[[x_col]].values
            y = numeric_df[y_col].values

            if reg_method == "linear":
                model = LinearRegression()
            elif reg_method == "polynomial":
                model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
            elif reg_method == "ridge":
                model = Ridge(alpha=1.0)
            elif reg_method == "lasso":
                model = Lasso(alpha=1.0)
            else:
                raise ValueError("Invalid regression method")

            model.fit(X, y)
            y_pred = model.predict(X)

            plt.figure(figsize=(8, 6))
            plt.scatter(X, y, label="Actual data", alpha=0.5)
            plt.plot(X, y_pred, color="red", label=f"{reg_method.capitalize()} Regression")
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"{reg_method.capitalize()} Regression: {x_col} vs {y_col}")
            plt.legend()
            plt.show()

            print(f"{reg_method.capitalize()} Regression: {x_col} vs {y_col}")
            if reg_method == "linear":
                print(f"Coefficient: {model.coef_[0]:.4f}")
                print(f"Intercept: {model.intercept_:.4f}")
            print(f"R-squared: {model.score(X, y):.4f}")
            
            # Calculer la valeur p pour la régression linéaire
            if reg_method == "linear":
                n = len(X)
                k = 1  # nombre de variables indépendantes
                dof = n - k - 1
                t_stat = model.coef_[0] / (np.sqrt(np.sum((y - y_pred) ** 2) / dof) / np.sqrt(np.sum((X - np.mean(X)) ** 2)))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof))
                print(f"p-value: {p_value:.4f}")
            
            print("\n")


def analyze_categorical_relationships(df, method="pearson"):
    """
    Analyze relationships between categorical variables using contingency tables and chi-square test.

    Parameters:
        df (pd.DataFrame): DataFrame containing categorical variables.
        method (str): Chi-square test method ('pearson', 'log-likelihood', 'freeman-tukey', 'mod-log-likelihood', 'neyman', 'cressie-read').

    Returns:
        None
    """
    categorical_df = df.select_dtypes(include=['object', 'category'])

    for i, col1 in enumerate(categorical_df.columns):
        for col2 in categorical_df.columns[i+1:]:
            # Contingency table
            cont_table = pd.crosstab(categorical_df[col1], categorical_df[col2])
            print(f"Contingency Table: {col1} vs {col2}")
            print(cont_table)
            print("\n")

            # Chi-square test
            chi2, p_value, dof, expected = chi2_contingency(cont_table, lambda_=method)
            print(f"Chi-square Test Results ({method} method):")
            print(f"Chi-square statistic: {chi2:.4f}")
            print(f"p-value: {p_value:.4f}")
            print(f"Degrees of freedom: {dof}")
            print("\n")

            # Visualize contingency table
            plt.figure(figsize=(10, 8))
            sns.heatmap(cont_table, annot=True, fmt='d', cmap='YlGnBu')
            plt.title(f"Contingency Table: {col1} vs {col2}")
            plt.show()
