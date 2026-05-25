"""
Data Preprocessing Module for Flight Cost Predictor
Handles data cleaning, feature engineering, and label encoding
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset and perform initial cleaning.

    Args:
        filepath: Path to the CSV dataset file

    Returns:
        Cleaned DataFrame
    """
    df = pd.read_csv(filepath)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Drop rows with null values
    df.dropna(inplace=True)

    return df


def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract useful features from date and time columns.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with extracted datetime features
    """
    # Extract departure hour and minute
    if 'Dep_Time' in df.columns:
        df['Dep_Hour'] = pd.to_datetime(df['Dep_Time']).dt.hour
        df['Dep_Minute'] = pd.to_datetime(df['Dep_Time']).dt.minute
        df.drop('Dep_Time', axis=1, inplace=True)

    # Extract arrival hour and minute
    if 'Arrival_Time' in df.columns:
        df['Arrival_Hour'] = pd.to_datetime(df['Arrival_Time']).dt.hour
        df['Arrival_Minute'] = pd.to_datetime(df['Arrival_Time']).dt.minute
        df.drop('Arrival_Time', axis=1, inplace=True)

    # Extract journey day and month from Date_of_Journey
    if 'Date_of_Journey' in df.columns:
        df['Journey_Day'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y').dt.day
        df['Journey_Month'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y').dt.month
        df.drop('Date_of_Journey', axis=1, inplace=True)

    return df


def process_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Duration column from string format to total minutes.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with Duration converted to minutes
    """
    if 'Duration' not in df.columns:
        return df

    def duration_to_minutes(duration):
        parts = duration.split()
        hours = 0
        minutes = 0
        for part in parts:
            if 'h' in part:
                hours = int(part.replace('h', ''))
            elif 'm' in part:
                minutes = int(part.replace('m', ''))
        return hours * 60 + minutes

    df['Duration_Minutes'] = df['Duration'].apply(duration_to_minutes)
    df.drop('Duration', axis=1, inplace=True)

    return df


def encode_categorical_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply label encoding to categorical columns.

    Args:
        df: Input DataFrame

    Returns:
        Tuple of (encoded DataFrame, dict of encoders)
    """
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Remove target column if present
    if 'Price' in categorical_cols:
        categorical_cols.remove('Price')

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def preprocess_pipeline(filepath: str) -> tuple[pd.DataFrame, dict]:
    """
    Full preprocessing pipeline from raw data to model-ready features.

    Args:
        filepath: Path to raw CSV data

    Returns:
        Tuple of (processed DataFrame, encoders dict)
    """
    df = load_and_clean_data(filepath)
    df = extract_datetime_features(df)
    df = process_duration(df)
    df, encoders = encode_categorical_features(df)

    return df, encoders


def get_feature_names(df: pd.DataFrame, target_col: str = 'Price') -> tuple[list, str]:
    """
    Get feature column names and target column name.

    Args:
        df: Processed DataFrame
        target_col: Name of target column

    Returns:
        Tuple of (feature names list, target column name)
    """
    features = [col for col in df.columns if col != target_col]
    return features, target_col
