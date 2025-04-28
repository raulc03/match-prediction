import pandas as pd
from functools import partial

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from xgboost import XGBClassifier


# Random State (for reproductibility)
rs = 69

# Parameters for cross-validation grid search
param_grid = [
    {
        # Logistic REgression parameters
        'clf': [LogisticRegression(max_iter=1000)],
        'clf__random_state': [rs],
        'clf__solver': ['liblinear'],  # Efficient solver for small dataset
        # L1 (Lasso) and L2 (Ridge) regularization
        'clf__penalty': ['l1', 'l2']
    },
    {
        # Support Vector Machine Parameters
        'clf': [SVC()],
        # Linear kernel for feature interpretability
        'clf__kernel': ['linear'],
        'clf__random_state': [rs],
    },
    {
        # Decision Tree Parameters
        'clf': [DecisionTreeClassifier()],
        'clf__random_state': [rs],
        # Gini impurity for split quality measurement
        'clf__criterion': ['gini'],
    },
    {
        # Random Forest parameters
        'clf': [RandomForestClassifier()],
        'clf__n_estimators': [50, 100, 200],  # Testing different forest sizes
        'clf__random_state': [rs],
    },
    {
        # XGBoost parameters
        'clf': [XGBClassifier()],
        'clf__learning_rate': [0.01, 0.1],  # Testing different learning rates
        # Binary classification objective
        'clf__objective': ['binary:logistic'],
        'clf__random_state': [rs],
    }
]


def classify(df: pd.DataFrame, num_discrete: list,
             num_continous: list, cat_cols: list):
    """
    Perform classfication on football match results using multiple models.

    This function builds a machine learning pipeline that:
    1. Splits data into training and testing sets
    2. Preprocesses different feature type appropriately
    3. Performs grid search cross-validation to find the best model
    4. Evaluates the best model on the test set

    Parameters
    ----------
    df : pd.DataFrame
       DataFrame containing match data and features
    num_discrete : list
        List of column names for discrete numerical features
    num_continous : list
        List of column names for continuous numerical features
    cat_cols : list
        List of column names for categorical features

    Returns
    -------
    None
         Results are printed to standard output
    """
    # Create a copy of the input data
    X = df.copy()

    # Extract the target variable (match result)
    y = X.pop('Result')

    # Split data into trainig (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=rs)

    # Define preprocessing steps for differents feature types
    preproc = ColumnTransformer([
        ('num_cont', StandardScaler(), num_continous),
        ('num_disc', StandardScaler(), num_discrete),
        ('cat', OneHotEncoder(handle_unknown='ignore'),
         cat_cols),
    ], remainder='drop')  # Ignore any columns not specified

    # Create a Pipeline with preprocessing and classifier
    pipe = Pipeline([
        ('preproc', preproc),
        # Default classifier (will be replaced by GridSearchCV)
        ('clf', LogisticRegression()),
    ])

    # Perform grid search cross-validation to find the best model
    grid = GridSearchCV(pipe, param_grid, cv=5)  # 5-fold cross-validation
    grid.fit(X_train, y_train)

    # Predict on test set using the best estimator
    y_pred = grid.best_estimator_.predict(X_test)

    print('Accuracy in validation:', accuracy_score(y_test, y_pred))
    print('Best Score in CV (Training):', grid.best_score_)
    print('Best estimator:',
          grid.best_estimator_.named_steps['clf'].__class__.__name__)


def classify_feature_selector(df: pd.DataFrame, num_discrete: list,
                              num_continous: list,
                              cat_cols: list) -> None:
    """
    Perform classification with feature selection on football match data.

    This function builds an advanced ML pipeline that:
    1. Splits data into training and testing sets
    2. Applies initial preprocessing for numerical and categorical features
    3. Selects the top 5 most informative features using mutual information
    4. Applies special scaling only to selected numerical features
    5. Performs grid search cross-validation to find the best model
    6. Evaluates the best model on the test set

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing match data and features
    num_discrete : list
        List of column names for discrete numerical features
    num_continous : list
        List of column names for continuous numerical features
    cat_cols : list
        List of column names for categorical features

    Returns
    -------
    None
        Results are printed to standard output

    Notes
    -----
    This function uses a more sophisticated pipeline than the basic classify()
    function, featuring feature selection and targeted scaling.
    """
    # Create a copy of the input data
    X = df.copy()

    # Extract the target variable (match result)
    y = X.pop('Result')

    # Split data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=rs)

    # Combine all numerical columns
    num_cols = num_continous + num_discrete

    # Define initial preprocessing steps
    preproc = ColumnTransformer([
        # Pass through numerical features without scaling
        # (scaling happens later)
        ('num', 'passthrough', num_cols),

        # One-hot encode categorical features
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ], remainder='drop')  # Ignore any columns not specified

    # Set up mutual information scoring for feature selection
    # with fixed random state
    mi_score = partial(mutual_info_classif, random_state=rs)

    # Create feature selector to select the top 5 most informative features
    feature_selector = SelectKBest(mi_score, k=5)

    # Create initial pipeline for preprocessing and feature selection
    pipe = Pipeline([
        ('preproc', preproc),
        ('feature_selector', feature_selector),
    ])

    # Fit the pipeline to get column names after one-hot encoding
    pipe.fit(X_train, y_train)

    # Get the categorical encoder from the pipeline
    cat_encoder = pipe.named_steps['preproc'].named_transformers_['cat']

    # Get the names of one-hot encoded categorical columns
    cat_names = cat_encoder.get_feature_names_out(cat_cols).tolist()

    # Combine all column names after preprocessing
    all_preproc_names = num_cols + cat_names

    # Get the feature selector from the pipeline
    selector = pipe.named_steps['feature_selector']

    # Get the boolean mask indicating which features were selected
    mask = selector.get_support()

    # Get the names of selected features using the mask
    selected_names = list(pd.Series(all_preproc_names)[mask])

    # Identify which selected features are numerical
    num_cols_selected = [
        col for col in selected_names
        if col.rsplit('_', maxsplit=1)[0] not in cat_cols]

    # Create a transformer to scale only the selected numerical features
    final_scaler = ColumnTransformer([
        ('scale_num', StandardScaler(), [
         selected_names.index(n) for n in num_cols_selected]),
    ], remainder='passthrough')  # Keep other features unchanged

    # Create the final pipeline with:
    # 1. Initial preprocessing
    # 2. Feature selection
    # 3. Selective scaling of numerical features
    # 4. Classification
    final_pipe = Pipeline([
        ('preproc', preproc),
        ('selector', selector),
        ('post_scale', final_scaler),
        # Default classifier (will be replaced by GridSearchCV)
        ('clf', LogisticRegression())
    ])

    # Perform grid search cross-validation to find the best model
    # 5-fold cross-validation
    grid = GridSearchCV(final_pipe, param_grid, cv=5)
    grid.fit(X_train, y_train)

    # Predict on test set using the best estimator
    y_pred = grid.best_estimator_.predict(X_test)

    # Print evaluation metrics
    print('Accuracy in validation:', accuracy_score(y_test, y_pred))
    print('Best Score in CV (Training):', grid.best_score_)
    print('Best estimator:',
          grid.best_estimator_.named_steps['clf'].__class__.__name__)
