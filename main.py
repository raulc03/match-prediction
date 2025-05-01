from data_cleaning import DataCleaning
from feature_engineer import FeatureEngineering
from classify import classify, classify_feature_selector
from utils import read_data


def main():
    """
    Execute the complete match prediction workflow for Liga 1 teams.

    This function orchestrates the entire data processing and
    classification pipeline:
    1. Loads the dataset of Liga 1 matches from 2020-2024
    2. Performs data cleaning operations
    3. Creates baseline features
    4. Runs classification using baseline features
    5. Augments the feature set with addtional engineered features
    6. Runs classification using the augmented feature set
    7. Performs feature selection and final classification

    Notes
    -----
    The features are organized into three categories:
    - Discrete numerical features (like goals scored/conceded)
    - Continuos numerical features ( like form-based metrics)
    - Categorical features (like round names)
    """
    # Path to the dataset containing Liga 1 match data
    path = 'data/Liga_1_Matches_2014-2024.csv'

    # Load the raw data from CSV file
    df = read_data(path)

    # Clean the data to handle missing values, etc
    df = DataCleaning(df).clean_data()

    print(df.shape)

    # Initiliaze feature engineering object with cleaned dataframe
    fe = FeatureEngineering(df)

    # Create baseline features for initial model evaluation
    df = fe.baseline_fe()

    # Define feature categories for the baseline model
    # HTGS = Home Team Goals Scored, ATGS = Away Team Goals Scored
    # HTGC = Home Team Goals Conceded, ATGC = Away Team Goals Conceded
    # HTGD = Home Team Goal Difference, ATGD = Away Team Goal Difference
    # MW = Match Week
    num_discrete = ['HTGS', 'ATGS', 'HTGC',
                    'ATGC', 'HTGD', 'ATGD', 'MW', 'Day']

    # Form-based features that track team performance over recent matches
    num_continuos = ['HTGSForm', 'ATGSForm', 'HTGCForm', 'ATGCForm']

    # Categorical features
    cat_cols = ['Round']

    # Execute classification with baseline features
    print('Baseline:')
    classify(df, num_discrete, num_continuos, cat_cols)

    # Create additional engineered features to improve model performance
    df = fe.augment_features()

    # Add new features to each category
    # HTP = Home Team Points, ATP = Home Team Points, PD = Points Difference
    num_discrete += ['HTP', 'ATP', 'PD', 'HTGS', 'ATGS', 'HTGC', 'ATGC']

    # Add form-based features for poitns
    num_continuos += ['HTPForm', 'ATPForm', 'PDForm']

    # Add last match results as categorical features
    # HLM = Home Last Match,
    # ALM = Away Last Match (1-5 represent previous matches)
    cat_cols += ['HLM_1', 'HLM_2', 'HLM_3', 'HLM_4', 'HLM_5',
                 'ALM_1', 'ALM_2', 'ALM_3', 'ALM_4', 'ALM_5']

    # Execute classification with augmented feature set
    print('\nAugmented:')
    classify(df, num_discrete, num_continuos, cat_cols)

    # Perform feature selection to identify most impactful features
    # and execute final classification with optimal feature subset
    print('\nFeature Selection:')
    classify_feature_selector(df, num_discrete, num_continuos, cat_cols)


if __name__ == '__main__':
    main()
