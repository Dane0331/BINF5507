# import all necessary libraries here
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Impute Missing Values
def impute_missing_values(data, strategy='mean'):
    """
    Fill missing values in the dataset.
    :param data: pandas DataFrame
    :param strategy: str, imputation method ('mean', 'median', 'mode')
    :return: pandas DataFrame with missing values filled
    """
    df = data.copy()
    # Identify numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    if strategy in ['mean', 'median']:
        # Build a dictionary for numeric columns based on the chosen strategy
        fill_values = {
            col: (df[col].mean() if strategy == 'mean' else df[col].median())
            for col in numeric_cols
        }
        # For non-numeric columns, fill with the mode
        df.fillna(value=fill_values, inplace=True)

    elif strategy == 'mode':
        # Use the mode for all columns
        fill_values = {col: df[col].mode()[0] for col in df.columns}
        df.fillna(value=fill_values, inplace=True)

    else:
        raise ValueError("Unsupported strategy. Use 'mean', 'median', or 'mode'.")

    for col in non_numeric_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df


# 2. Remove Duplicates
def remove_duplicates(data):
    """
    Remove duplicate rows from the dataset.
    :param data: pandas DataFrame
    :return: pandas DataFrame
    """
    # Remove duplicate rows
    df = data.copy()
    df.drop_duplicates(inplace=True)
    return df


# 3. Normalize Numerical Data
def normalize_data(data,method='minmax'):
    """Apply normalization to numerical features.
    :param data: pandas DataFrame
    :param method: str, normalization method ('minmax' (default) or 'standard')
    """
    # Normalize numerical data using Min-Max or Standard scaling
    df = data.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Normalization method not in function. Use 'minmax' or 'standard' instead.")

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

# 4. Remove Redundant Features   
def remove_redundant_features(data, threshold=0.9):
    """Remove redundant or duplicate columns.
    :param data: pandas DataFrame
    :param threshold: float, correlation threshold
    :return: pandas DataFrame
    """
    # Remove redundant features based on the correlation threshold (HINT: you can use the corr() method)
    df = data.copy()
    # Numeric cols only for computing correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr().abs()
    
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
    df.drop(columns=to_drop, inplace=True)
    return df
    
# ---------------------------------------------------

def simple_model(input_data, split_data=True, scale_data=False, print_report=False):
    """
    A simple logistic regression model for target classification.
    Parameters:
    input_data (pd.DataFrame): The input data containing features and the target variable 'target' (assume 'target' is the first column).
    split_data (bool): Whether to split the data into training and testing sets. Default is True.
    scale_data (bool): Whether to scale the features using StandardScaler. Default is False.
    print_report (bool): Whether to print the classification report. Default is False.
    Returns:
    None
    The function performs the following steps:
    1. Removes columns with missing data.
    2. Splits the input data into features and target.
    3. Encodes categorical features using one-hot encoding.
    4. Splits the data into training and testing sets (if split_data is True).
    5. Scales the features using StandardScaler (if scale_data is True).
    6. Instantiates and fits a logistic regression model.
    7. Makes predictions on the test set.
    8. Evaluates the model using accuracy score and classification report.
    9. Prints the accuracy and classification report (if print_report is True).
    """

    df = input_data.copy()
    # if there's any missing data, remove the columns
    input_data.dropna(axis=1, inplace=True)
    
    # Ensure Target Variable is Binary
    target = df.iloc[:, 0]
    target = target.astype(int) 
    
    if target.nunique() > 2:
        raise ValueError("Target column has more than 2 unique values. Logistic regression requires a binary classification target.")

    # Convert Categorical Variables to One-Hot Encoding
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Separate Features and Target
    features = df.iloc[:, 1:]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

    # Normalize Data (if requested)
    if scale_data:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Train Logistic Regression Model
    model = LogisticRegression(random_state=42, max_iter=100, solver='liblinear', penalty='l2', C=1.0)
    model.fit(X_train, y_train)

    # Make Predictions and Evaluate Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')

    if print_report:
        print('Classification Report:')
        print(report)
        print('Read more about the classification report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html and https://www.nb-data.com/p/breaking-down-the-classification')
    

    return model