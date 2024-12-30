# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.stats.api as sms
from statsmodels.compat import lzip

# Section 1: Load dataset and initial exploration
def load_and_explore_data():
    """Load dataset, perform initial exploration, and clean the data."""
    df = pd.read_csv('https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/001/839/original/Jamboree_Admission.csv')
    df.rename(columns={'LOR ': 'LOR', 'Chance of Admit ': 'Chance of Admit'}, inplace=True)
    df.drop('Serial No.', axis=1, inplace=True)
    print("Dataset Info:")
    df.info()
    print("Dataset Description:")
    print(df.describe())
    print("Missing Values:")
    print(df.isna().sum())
    return df

# Section 2: Exploratory Data Analysis
def perform_eda(df):
    """Perform exploratory data analysis and visualize key trends."""
    list1 = ['GRE Score', 'TOEFL Score', 'SOP', 'LOR', 'CGPA', 'Chance of Admit']

    # Grouping by University Rating
    result = df.groupby('University Rating')[list1].mean()
    result.columns = [f"avg_{i}" for i in result.columns]
    print("Average values by University Rating:")
    print(result.round(2).reset_index())

    # Grouping by Research Experience
    result = df.groupby('Research')[list1].mean()
    result.columns = [f"avg_{i}" for i in result.columns]
    print("Average values by Research Experience:")
    print(result.round(2).reset_index())

    # Visualizing distributions
    for i in list1:
        plt.figure()
        sns.histplot(df[i], bins=25, kde=True, color='purple')
        plt.title(f"Distribution of {i}")

    # Boxplots to check for outliers
    for i in list1[:-1]:
        plt.figure()
        sns.boxplot(df[i], color='yellow')
        plt.title(f"Boxplot of {i}")

    # Correlation Heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(df.corr(), annot=True, cmap='YlGnBu')
    plt.title("Correlation Heatmap")

    # Scatterplots for predictor vs target variable
    for i in list1[:-1]:
        plt.figure()
        sns.scatterplot(x=df[i], y=df['Chance of Admit'], legend=True)
        plt.title(f"Scatterplot: {i} vs Chance of Admit")

# Section 3: Data Preprocessing
def preprocess_data(df):
    """Prepare the dataset for modeling."""
    X = df.drop('Chance of Admit', axis=1)
    y = df['Chance of Admit']

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return pd.DataFrame(X_train_scaled, columns=X_train.columns), pd.DataFrame(X_test_scaled, columns=X_train.columns), y_train, y_test

# Section 4: Model Training and Evaluation
def train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test):
    """Train models and evaluate their performance."""
    # Linear Regression
    model = LinearRegression()
    score = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"Mean R-squared for Linear Regression: {np.mean(score)}")

    # Regularization Techniques (Lasso and Ridge)
    alpha = np.linspace(0.0, 0.1, 10)
    r2_lasso = []
    r2_ridge = []
    for i in alpha:
        model_lasso = Lasso(alpha=i)
        r2_lasso.append(np.mean(cross_val_score(model_lasso, X_train_scaled, y_train, cv=5, scoring='r2')))
        
        model_ridge = Ridge(alpha=i)
        r2_ridge.append(np.mean(cross_val_score(model_ridge, X_train_scaled, y_train, cv=5, scoring='r2')))

    # Visualizing Lasso and Ridge performance
    plt.figure()
    sns.lineplot(x=alpha, y=r2_lasso, label='Lasso')
    sns.lineplot(x=alpha, y=r2_ridge, label='Ridge')
    plt.title("Lasso vs Ridge Performance")

    # Model Evaluation Metrics
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    print(f"MSE for Linear Regression: {np.sqrt(mean_squared_error(y_test, predictions))}")

    # Feature Importance
    coeffs = model.coef_
    columns_X = X_train_scaled.columns
    plt.figure(figsize=(12, 6))
    sns.barplot(x=columns_X, y=coeffs)
    plt.title("Feature Importance")

    # Statistical Analysis
    X_train_scaled = sm.add_constant(X_train_scaled)
    model_0 = sm.OLS(y_train.values, X_train_scaled).fit()
    print(model_0.summary())

    # Residual Analysis
    residuals = y_test - predictions
    sns.scatterplot(x=predictions, y=residuals)
    plt.title('Residuals vs Fitted Values')

    # Testing for Homoscedasticity
    name = ['F statistic', 'p-value']
    test = sms.het_goldfeldquandt(residuals, X_test_scaled)
    print(dict(zip(name, test)))

    # Normality of Residuals
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Distribution')

    # Final Model Evaluation
    print(f"R2 score for final model: {r2_score(y_test, predictions)}")

# Main execution
if __name__ == "__main__":
    data = load_and_explore_data()
    perform_eda(data)
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(data)
    train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test)