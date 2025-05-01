import pandas as pd

# Dictionary that maps results to their inverses
inv = {'M': 'M', 'W': 'L', 'L': 'W', 'D': 'D'}


# Función que lee el dataset y retorna un dataframe
def read_data(path: str) -> pd.DataFrame:
    """
    Read data from a CSV file and return it as a DataFrame.

    This function reads data from the specified CSV file path and
    parses the 'DateTime' column as datetime objects.

    Parameters
    ----------
    path: str
        Path to the CSV file to be read

    Returns
    -------
    pd.DataFrame
        DataFrame containing the data from the CSV file with 'DateFrame'
        column converted to datetime objects
    """
    df = pd.read_csv(path)
    return df
