import pandas as pd
from utils import inv


class DataCleaning:
    """
    A class for cleaning and preprocessing football match data.

    This class handles various data cleaning operations for football match
    datasets, including fixing team name inconsistencies, handling match data,
    fixing format issues, and removing unnecessary columns.

    Attributes
    ----------
    df: pd.DataFrame
        The DataFrame containing match data to be cleaned

    Methods
    -------
    clean_data() -> pd.DataFrame
        Execute the complete data cleaning pipeline and return
        the cleaned DataFrame
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the DataCleaning class with a DataFrame.

        Parameters
        ----------
        df: pd.DataFrame
            The DataFrame containing raw match data
        """
        self.df: pd.DataFrame = df.copy()

    def clean_data(self) -> pd.DataFrame:
        """
        Execute the complete data cleaning pipeline.

        This method runs all private cleaning methods in the appropiate
        sequence to fully process the raw data.

        Returns
        -------
        pd.DataFrame
            The cleaned and processed DataFrame
        """
        self.__fix_teams_names()
        self.__get_matches()
        self.__fix_gf_ga_format()
        self.__fix_rounds()
        self.__drop_columns()

        return self.df

    def __drop_columns(self) -> None:
        """
        Remove unnecessary columns from the DataFrame.

        This method drops columns that are not needed for the analysis,
        such as metada and redundant information.
        """
        df: pd.DataFrame = self.df.copy()

        # List of columns that don't contribute to the analysis
        useless_cols = ['Unnamed: 0', 'Comp', 'Venue', 'Team', 'Opponent',
                        'Date', 'Time', 'Captain', 'Formation',
                        'Opp Formation', 'Referee', 'Day']
        df.drop(useless_cols, axis=1, inplace=True)

        self.df = df

    def __fix_gf_ga_format(self) -> None:
        """
        Fix the format of goals scored and conceded.

        This method removes penalty shootout information from goal data
        and converts the values to integers.
        """
        df: pd.DataFrame = self.df.copy()

        def ignore_penalties(row: pd.Series) -> pd.Series:
            """
            Remove penalty shootout results from goal data.

            Parameters
            ----------
            row : pd.Series
                A row from the DataFrame

            Returns
            -------
            pd.Series
                The modified row with clean goal data
            """
            row['GF'] = row['GF'].split('(')[0].strip()  # type:ignore
            row['GA'] = row['GA'].split('(')[0].strip()  # type:ignore
            return row

        df = df.apply(ignore_penalties, axis=1)  # type: ignore

        # Convert goal columns to integer type
        df['GF'] = df['GF'].astype(int)
        df['GA'] = df['GA'].astype(int)

        self.df = df

    def __fix_teams_names(self) -> None:
        """
        Standardize round names in the competition.

        This method replaces inconsistent team names with their
        standardized versions using a predefined mapping.
        """
        df: pd.DataFrame = self.df.copy()

        # Dictionary mapping inconsistent team names to standardized names
        replace = {'Alianza Univ': 'Alianza Universidad',
                   'Universidad Técnica de Cajamarca': 'UTC'}

        df['Team'] = df['Team'].replace(replace)
        df['Opponent'] = df['Opponent'].replace(replace)

        self.df = df

    def __fix_rounds(self) -> None:
        """
        Standardize round names in the competition.

        This method ensures consistent naming conventions
        for tournament rounds.
        """
        df: pd.DataFrame = self.df.copy()

        df['Round'] = df['Round'].replace({'Finals': 'Final'})

        self.df = df

    def __get_matches(self) -> None:
        """
        Structure the data to have consistent home and away team format.

        This method:
        1. Filters for Liga 1 matches only
        2. Ensures each match has proper home/away team designation
        3. Adjusts goals and results accordingly
        4. Removes duplciates and sorts by date
        """
        df: pd.DataFrame = self.df.copy()

        df = df[df['Comp'] == 'Liga 1']  # type: ignore

        def parse_matches(row) -> pd.Series:
            """
            Standardize match records to follow a home/away team format.

            For away matches, this swaps team positions and adjusts
            goals and results accordingly.

            Parameters
            ----------
            row : pd.Series
                A row from the DataFrame

            Returns
            -------
            pd.Series
                The modified row with standardized match format
            """
            if row['Venue'] == 'Home':
                row['Home Team'] = row['Team']
                row['Away Team'] = row['Opponent']
            else:
                row['Home Team'] = row['Opponent']
                row['Away Team'] = row['Team']
                # Swap goals for away matches
                row['GF'], row['GA'] = row['GA'], row['GF']
                # Invert result for away matches
                row['Result'] = inv[row['Result']]

            return row

        df = (df.apply(parse_matches, axis=1)  # type: ignore
              # Remove duplicate matches
              .drop_duplicates(subset=['DateTime', 'Home Team', 'Away Team'],
                               ignore_index=True)
              # Sort matches by date and time
              .sort_values(['DateTime'], axis=0, ignore_index=True)
              )
        self.df = df
