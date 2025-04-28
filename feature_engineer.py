import pandas as pd
from utils import inv


class FeatureEngineering:
    """
    Class for applying baseline and advanced feature engineering techniques
    to a football match dataset

    This class generates new features based on match results to enhance
    the predictive power of the dataset for machine learning tasks.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the FeatureEngineering class with a copy
        of the input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing match data
        """
        self.df = df.copy()

    def baseline_fe(self) -> pd.DataFrame:
        """
        Perform baseline feature engineering.

        This method generates initial features such as match week,
        goals scored, goal difference, and recent team form.
        It also creates a binary target variable for classification tasks.

        Returns
        -------
        pd.DataFrame
            DataFrame with baseline features added.
        """
        self.__get_match_week()
        self.__goals_scored()
        self.__goals_difference()
        self.__teams_form()

        self.df_o = self.df.copy()

        # Create binary target feature for model training
        self.__binary_target()

        return self.df

    def augment_features(self) -> pd.DataFrame:
        """
        Perform additional feature engineering on top of baselien features.

        This method andds advanced features such as cumulative points,
        streaks, and results from the last 5 matches for each team.

        Returns
        -------
        pd.DataFrame
            DataFrame with additional features added
        """
        self.df = self.df_o.copy()

        # Add features realted to poitns acumulation and streaks
        self.__get_points()
        self.__streaks()
        self.__get_5_last_games()

        self.df_o = self.df.copy()

        # Rebuild binary target feature after feature augmentation
        self.__binary_target()

        return self.df

    def __get_5_last_games(self) -> None:
        """
        Obtain the results of the last 5 matches
        for each team before each match.

        This method generates columns LM_1 to LM_5
        for both home and away teams, representing
        the last matches' results prior to the current match.

        Returns
        -------
        None
            Updates self.df with columns LM_1 to LM_5
            for both home and away teams.
        """
        df = self.df.copy()

        df['year'] = df['DateTime'].dt.year

        df_home = df[['year', 'Home Team', 'Result']].rename(  # type:ignore
            columns={'Home Team': 'Team'})
        df_home['Venue'] = 'home'

        df_away = df[['year', 'Away Team', 'Result']].rename(  # type:ignore
            columns={'Away Team': 'Team'})
        df_away['Venue'] = 'away'

        # Merge home and away match data
        df_large = pd.concat([df_home, df_away]).sort_index()

        # GEnerate last 5 matches result features
        for i in range(1, 6):
            colname = 'LM_' + str(i)
            df_large[colname] = (df_large.groupby(['year', 'Team'])
                                 .shift(i, fill_value='M')
                                 .apply(lambda row: row['Result']
                                        if row['Venue'] == 'home'
                                        else inv[row['Result']],
                                        axis=1))

            df_pivot = df_large.pivot(columns='Venue', values=colname)
            df_pivot.columns = ['A' + colname, 'H' + colname]

            # Add features for both home and away teams
            df = pd.concat(
                [df, df_pivot[['H' + colname, 'A' + colname]]], axis=1)

        self.df = df.drop('year', axis=1)

    def __get_points(self) -> None:
        """
        Calculate cumulative points for home and away teams.

        This method computes points gained before each match, based on
        match results, and generates additional features such as
        points difference and form ratios.
        """

        df = self.df.copy()
        df['year'] = df['DateTime'].dt.year

        cols = ['year', 'Home Team', 'Result']
        df_home = df[cols].rename(columns={'Home Team': 'Team'})  # type:ignore
        df_home['venue'] = 'Home'

        cols = ['year', 'Away Team', 'Result']
        df_away = df[cols].rename(
            columns={'Away Team': 'Team'})  # type: ignore
        df_away['venue'] = 'Away'

        df_large = pd.concat([df_home, df_away])
        df_large = df_large.sort_index()

        def which_points(row: pd.Series) -> int:
            """
            Calculate points based on match result and venue.

            Parameter
            ---------
            row : pd.Series
                A row containing match result and venue information

            Returns
            -------
            int
                POints earned: 3 for a win, 1 for a draw, 0 for a loss.
            """
            result = row['Result']
            venue = row['venue']

            if result == 'D':
                return 1
            if venue == 'Home':
                if result == 'W':
                    return 3
            else:
                if result == 'L':
                    return 3

            return 0

        # Calculate points per match
        df_large['Points'] = df_large.apply(which_points, axis=1)

        # Calculate cumulative points
        df_large['AccPoints'] = (df_large
                                 .groupby(['year', 'Team'])['Points']
                                 .shift(fill_value=0)
                                 .groupby([df_large['year'], df_large['Team']])
                                 .cumsum())
        df_pivot = df_large.pivot(columns='venue', values='AccPoints')
        df_pivot.columns = ['ATP', 'HTP']

        # Merge cumulative points into main dataframe
        df = (pd.concat([df, df_pivot[['HTP', 'ATP']]], axis=1)
              .drop('year', axis=1)
              )

        # Calculate points difference
        df['PD'] = df['HTP'] - df['ATP']

        # Calculate average points per match for form assessment
        df['HTPForm'] = df['HTP'] / (df['MW'] - 1)
        df['ATPForm'] = df['ATP'] / (df['MW'] - 1)

        # Fill missing values with 0 (beginning of season)
        df['HTPForm'] = df['HTPForm'].fillna(0)
        df['ATPForm'] = df['ATPForm'].fillna(0)

        # Calculate points form difference
        df['PDForm'] = df['HTPForm'] - df['ATPForm']

        self.df = df

    def __streaks(self) -> None:
        """
         Calculates unbeaten and losing streaks for home and away teams.

            This function adds the following columns to the main DataFrame:
                - HTUS: Home Team Unbeaten Streak
                - ATUS: Away Team Unbeaten Streak
                - HTLS: Home Team Losing Streak
                - ATLS: Away Team Losing Streak

        Note
        ----
            Updates the `self.df` attribute by adding the calculated
            streak columns.
        """

        def calc_unbeaten_streaks(results) -> list:
            u_streak = 0
            unbeaten_list = []

            for res in results:
                unbeaten_list.append(u_streak)

                if res != 'L':
                    u_streak += 1
                else:
                    u_streak = 0

            return unbeaten_list

        def calc_losing_streaks(results) -> list:
            l_streak = 0
            losing_list = []

            for res in results:
                losing_list.append(l_streak)

                if res != 'L':
                    l_streak = 0
                else:
                    l_streak += 1

            return losing_list

        df = self.df.copy()

        df_home = df[['Home Team', 'Result']].rename(  # type:ignore
            columns={'Home Team': 'Team'})
        df_home['Venue'] = 'home'

        df_away = df[['Away Team', 'Result']].rename(  # type:ignore
            columns={'Away Team': 'Team'})
        df_away['Venue'] = 'away'

        df_large = pd.concat([df_home, df_away])
        df_large = df_large.sort_index()

        df_large['Result Team'] = df_large.apply(
            lambda r: r['Result'] if r['Venue'] == 'home'
            else inv[r['Result']],
            axis=1)

        df_large['US'] = (df_large
                          .groupby('Team')['Result Team']
                          .transform(lambda serie:
                                     calc_unbeaten_streaks(serie))
                          )
        df_large['LS'] = (df_large
                          .groupby('Team')['Result Team']
                          .transform(lambda serie: calc_losing_streaks(serie))
                          )

        df_pivot_us = df_large.pivot(columns='Venue', values='US')
        df_pivot_us.columns = ['ATUS', 'HTUS']
        df_pivot_ls = df_large.pivot(columns='Venue', values='LS')
        df_pivot_ls.columns = ['ATLS', 'HTLS']

        self.df = (pd.concat([df, df_pivot_us[['HTUS', 'ATUS']],
                              df_pivot_ls[['HTLS', 'ATLS']]], axis=1))

    def __binary_target(self) -> None:
        """
        Converts the 'Result' column into a binary target:
            - 1 for a win ('W')
            - 0 for any other outcome.

        Note
        ----
            Updates the `self.df` attribute by converting the 'Result' column
        """
        df = self.df.copy()

        df['Result'] = df['Result'].apply(
            lambda result: 1 if result == 'W' else 0)

        self.df = df

    def __teams_form(self) -> None:
        """
        Calculates the form of home and away teams based on goals
        scored and conceded relative to matches played up to that date.

        This function adds the following columns:
            - HTGSForm: Home Team Goals Scored Form
            - ATGSForm: Away Team Goals Scored Form
            - HTGCForm: Home Team Goals Conceded Form
            - ATGCForm: Away Team Goals Conceded Form

        Note
        ----
            Updates the `self.df` attribute by adding team form columns.
        """
        df = self.df.copy()
        df['HTGSForm'] = df['HTGS'] / (df['MW'] - 1)
        df['ATGSForm'] = df['ATGS'] / (df['MW'] - 1)

        df['HTGSForm'] = df['HTGSForm'].fillna(0)
        df['ATGSForm'] = df['ATGSForm'].fillna(0)

        df['HTGCForm'] = df['HTGC'] / (df['MW'] - 1)
        df['ATGCForm'] = df['ATGC'] / (df['MW'] - 1)

        df['HTGCForm'] = df['HTGCForm'].fillna(0)
        df['ATGCForm'] = df['ATGCForm'].fillna(0)

        self.df = df

    def __get_match_week(self):
        """
        Calculates the match week number considering breaks between games.

        Adds a 'MW' column indicating the match week,
        determined by taking the maximum number of games played
        between the home and away teams at each date.

        Note
        ----
            Updates the `self.df` attribute by adding the
            'MW' (Match Week) column.
        """

        df = self.df.copy()

        df['year'] = df['DateTime'].dt.year

        cols = ['year', 'Home Team']
        df_home = df[cols].rename(columns={'Home Team': 'Team'})  # type:ignore
        df_home['venue'] = 'Home'

        cols = ['year', 'Away Team']
        df_away = df[cols].rename(columns={'Away Team': 'Team'})  # type:ignore
        df_away['venue'] = 'Away'

        df_largo = pd.concat([df_home, df_away])
        df_largo = df_largo.sort_index()

        df_largo['MW'] = (df_largo.groupby(['year', 'Team']).cumcount() + 1)
        df_pivot = df_largo.pivot(
            columns='venue', values='MW')

        df_pivot['MW'] = df_pivot[[
            'Away', 'Home']].max(axis=1)

        self.df = (pd.concat([df, df_pivot['MW']], axis=1)
                   .drop('year', axis=1))

    def __goals_difference(self):
        """
        Calculates the goal difference for home and away teams
        prior to each match.

        This function adds the following columns:
            - HTGD: Home Team Goal Difference
            - ATGD: Away Team Goal Difference

        Note
        ----
            Updates the `self.df` attribute by adding goal difference columns.
        """
        df = self.df.copy()

        df['HTGD'] = df['HTGS'] - df['HTGC']
        df['ATGD'] = df['ATGS'] - df['ATGC']

        self.df = df

    def __goals_scored(self) -> None:
        """
        Calculates the cumulative goals scored and conceded by
        home and away teams throughout the season.

        This function adds the following columns:
            - HTGS: Home Team Goals Scored
            - ATGS: Away Team Goals Scored
            - HTGC: Home Team Goals Conceded
            - ATGC: Away Team Goals Conceded

        Note
        ----
            Updates the `self.df` attribute by adding cumulative goals columns.
        """
        df = self.df.copy()

        # Columna útil para la agrupación por temporadas
        df['year'] = df['DateTime'].dt.year

        # DataFrame con los goles de los equipos locales
        # en sus respectivos años
        cols = ['year', 'Home Team', 'GF', 'GA']
        df_home = df[cols].rename(columns={'Home Team': 'Team',  # type:ignore
                                           'GF': 'Goals_scored',
                                           'GA': 'Goals_Conceded'})
        df_home['Venue'] = 'Home'

        # DataFrame con los goles de los equipo visitantes
        # en sus respectivos años
        cols = ['year', 'Away Team', 'GA', 'GF']
        df_away = df[cols].rename(columns={'Away Team': 'Team',  # type: ignore
                                           'GA': 'Goals_scored',
                                           'GF': 'Goals_Conceded'})
        df_away['Venue'] = 'Away'

        # Concatenamos para tenerlos todos en un solo DataFrame largo
        df_large = pd.concat([df_home, df_away])

        # Ordenamos por índice para que los partidos estén ordenados
        df_large = df_large.sort_index()

        # Creamos el dataframe de goles acumulados
        df_large['GS'] = (df_large
                          # Agrupamos mediante el año y el equipo y
                          # obtenemos la columna 'Goals Scored'
                          .groupby(['year', 'Team'])['Goals_scored']
                          # Shifteamos en 1 periodo para que la suma
                          # acumulativa comience en 0
                          .shift(fill_value=0)
                          # Volvemos a agrupar debido a que pandas
                          # olvida que había agrupado despues de shiftear
                          .groupby([df_large['year'], df_large['Team']])
                          # Realizamos la suma acumulativa por
                          # grupo (año y equipo)
                          .cumsum())

        df_large['GC'] = (df_large
                          .groupby(['year', 'Team'])['Goals_Conceded']
                          .shift(fill_value=0)
                          .groupby([df_large['year'], df_large['Team']])
                          .cumsum())

        # Pivoteamos para que los valores de la columna Venue (Away y Home),
        # sean las columnas y que la data de cada fila sean los
        # goles acumulados (GS). El orden será en base al índice (partido)
        df_pivot_gs = df_large.pivot(
            columns='Venue', values='GS')
        df_pivot_gc = df_large.pivot(
            columns='Venue', values='GC')

        df_pivot_gs.columns = ['ATGS', 'HTGS']
        df_pivot_gc.columns = ['ATGC', 'HTGC']
        df_pivot_gs = df_pivot_gs[['HTGS', 'ATGS']]
        df_pivot_gc = df_pivot_gc[['HTGC', 'ATGC']]

        # Concatemamos (ordenando las columnas) y retornamos el nuevo DataFrame
        self.df = (pd.concat([df, df_pivot_gs, df_pivot_gc], axis=1)
                   .drop('year', axis=1)
                   )
