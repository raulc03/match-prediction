import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class DataCleaning:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df: pd.DataFrame = df.copy()

    def clean_data(self) -> pd.DataFrame:
        self.__fix_teams_names()
        self.__get_matches()
        self.__fix_gf_ga_format()
        self.__fix_rounds()
        self.__drop_columns()

        return self.df

    def __drop_columns(self) -> None:
        df: pd.DataFrame = self.df.copy()

        useless_cols = ['Unnamed: 0', 'Comp', 'Venue', 'Team', 'Opponent',
                        'Date', 'Time', 'Captain', 'Formation',
                        'Opp Formation', 'Referee', 'Day']

        df.drop(useless_cols, axis=1, inplace=True)

        self.df = df

    def __fix_gf_ga_format(self) -> None:
        df: pd.DataFrame = self.df.copy()

        def ignore_penalties(row) -> pd.Series:
            row['GF'] = row['GF'].split('(')[0].strip()
            row['GA'] = row['GA'].split('(')[0].strip()
            return row

        df = df.apply(ignore_penalties, axis=1)  # type: ignore

        df['GF'] = df['GF'].astype(int)
        df['GA'] = df['GA'].astype(int)

        self.df = df

    def __fix_teams_names(self) -> None:
        df: pd.DataFrame = self.df.copy()

        # NOTE: Diccionario con las inconsistencias encontradas
        # NOTE: Puede aumentar mientras se agreguen más temporadas
        replace = {'Alianza Univ': 'Alianza Universidad',
                   'Universidad Técnica de Cajamarca': 'UTC'}

        df['Team'] = df['Team'].replace(replace)
        df['Opponent'] = df['Opponent'].replace(replace)

        self.df = df

    def __fix_rounds(self) -> None:
        df: pd.DataFrame = self.df.copy()

        df['Round'] = df['Round'].replace({'Finals': 'Final'})

        self.df = df

    def __get_matches(self) -> None:
        df: pd.DataFrame = self.df.copy()

        df = df[df['Comp'] == 'Liga 1']  # type: ignore

        def parse_matches(row) -> pd.Series:
            if row['Venue'] == 'Home':
                row['Home Team'] = row['Team']
                row['Away Team'] = row['Opponent']
            else:
                row['Home Team'] = row['Opponent']
                row['Away Team'] = row['Team']
                row['GF'], row['GA'] = row['GA'], row['GF']

                inv = {'W': 'L', 'L': 'W', 'D': 'D'}
                row['Result'] = inv[row['Result']]

            return row

        df = (df.apply(parse_matches, axis=1)  # type: ignore
              .drop_duplicates(subset=['DateTime', 'Home Team', 'Away Team'],
                               ignore_index=True)
              .sort_values(['DateTime'], axis=0, ignore_index=True)
              )

        self.df = df


# TODO: Feature Engineering (Fuera del pipeline)

class FeatureEngineering:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    def baseline_fe(self) -> pd.DataFrame:
        self.__get_match_week()
        self.__goals_scored_conceded()
        self.__goals_difference()
        self.__teams_form()
        self.__binary_target()

        return self.df

    # TODO: Agregar las demás features
    def augment_features(self) -> pd.DataFrame:
        self.__get_points()

        return self.df

    def __get_points(self) -> None:
        """Función que calcula los puntos con los que el equipo
        local y visitante llegan a cada partido.

        Parámetro
        ---------
        df: DataFrame con los datos para calcular los puntos

        Retorna
        -------
        df: DataFrame con las columnas HTP (Home Team Points) y
        ATP (Away Team Points)
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

        def which_points(row) -> int:
            """Función que permite calcular los puntos dependiendo de la
            localidad del equipo y el resultado del partido

            Parámetro
            --------
            row: Fila con los datos para realizar el cálculo

            Retorna
            -------
            punto: 3 - Equipo gana, 1 - Equipo empata, 0 - Equipo pierde
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

        # Calculamos el punto de cada equipo que obtiene del partido
        df_large['Points'] = df_large.apply(which_points, axis=1)

        # Calculamos el acumulado con un periodo de retraso para que comiencen
        # todas las temporadas con 0
        df_large['AccPoints'] = (df_large
                                 .groupby(['year', 'Team'])['Points']
                                 .shift(fill_value=0)
                                 .groupby([df_large['year'], df_large['Team']])
                                 .cumsum()
                                 )
        df_pivot = df_large.pivot(columns='venue', values='AccPoints')
        df_pivot.columns = ['ATP', 'HTP']

        df = (pd.concat([df, df_pivot[['HTP', 'ATP']]], axis=1)
              .drop('year', axis=1)
              )

        df['PD'] = df['HTP'] - df['ATP']

        self.df = df

    def __binary_target(self) -> None:
        df = self.df.copy()

        df['Result'] = df['Result'].apply(
            lambda result: 1 if result == 'W' else 0)

        self.df = df

    def __teams_form(self) -> None:
        df = self.df.copy()

        # Rendimiento del equipo local y visitante con
        # respectos a los goles marcados
        df['HTGSForm'] = df['HTGS'] / (df['MW'] - 1)
        df['ATGSForm'] = df['ATGS'] / (df['MW'] - 1)

        df['HTGSForm'] = df['HTGSForm'].fillna(0)
        df['ATGSForm'] = df['ATGSForm'].fillna(0)

        # Rendimiento del equipo local y visitante
        # con respecto a los goles recibidos
        df['HTGCForm'] = df['HTGC'] / (df['MW'] - 1)
        df['ATGCForm'] = df['ATGC'] / (df['MW'] - 1)

        df['HTGCForm'] = df['HTGCForm'].fillna(0)
        df['ATGCForm'] = df['ATGCForm'].fillna(0)

        self.df = df

    def __get_match_week(self):
        """Función que calcula la semana de partido de cada partido
            Parámetro
            ---------
            df: DataFrame con los datos a calcular
            Retorna
            -------
            DataFrame: Mismo DataFrame con la columna 'MW' (Match Week) añadida
        """
        df = self.df.copy()

        df['year'] = df['DateTime'].dt.year

        cols = ['year', 'Home Team']
        df_home = df[cols].rename(columns={'year': 'year',  # type: ignore
                                           'Home Team': 'Team'})
        df_home['venue'] = 'Home'

        cols = ['year', 'Away Team']
        df_away = df[cols].rename(columns={'year': 'year',  # type: ignore
                                           'Away Team': 'Team'})
        df_away['venue'] = 'Away'

        df_largo = pd.concat([df_home, df_away])
        df_largo = df_largo.sort_index()

        # Numeramos cada aparición de cada grupo desde 1
        df_largo['MW'] = (df_largo.groupby(['year', 'Team'])
                          .cumcount() + 1
                          )
        # Obtenemos el match week del equipo local y visitante
        df_pivot = df_largo.pivot(columns='venue', values='MW')

        # Creamos la columna 'MW' con el valor más alto de cada fila
        # Porque la diferencia puede deberse al descanso
        # de un equipo en una fecha
        # Pero la semana se jugó por lo que el retraso
        # de partidos jugados no se debe contar
        df_pivot['MW'] = df_pivot[['Away', 'Home']].max(axis=1)

        self.df = (pd.concat([df, df_pivot['MW']], axis=1)
                   .drop('year', axis=1))

    def __goals_difference(self):
        df = self.df.copy()

        # Home Team Goals Difference
        df['HTGD'] = df['HTGS'] - df['HTGC']
        # Away Team Goals Difference
        df['ATGD'] = df['ATGS'] - df['ATGC']

        self.df = df

    def __goals_scored_conceded(self) -> None:
        df = self.df.copy()

        # Columna útil para la agrupación por temporadas
        df['year'] = df['DateTime'].dt.year

        # DataFrame con los goles de los equipos locales
        # en sus respectivos años
        cols = ['year', 'Home Team', 'GF', 'GA']
        df_home = df[cols].rename(columns={'year': 'year',  # type: ignore
                                           'Home Team': 'Team',
                                           'GF': 'Goals_scored',
                                           'GA': 'Goals_Conceded'})
        df_home['Venue'] = 'Home'

        # DataFrame con los goles de los equipo visitantes
        # en sus respectivos años
        cols = ['year', 'Away Team', 'GA', 'GF']
        df_away = df[cols].rename(columns={'year': 'year',  # type: ignore
                                           'Away Team': 'Team',
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
        df_pivot_gs = df_large.pivot(columns='Venue', values='GS')
        df_pivot_gc = df_large.pivot(columns='Venue', values='GC')

        df_pivot_gs.columns = ['ATGS', 'HTGS']
        df_pivot_gc.columns = ['ATGC', 'HTGC']
        df_pivot_gs = df_pivot_gs[['HTGS', 'ATGS']]
        df_pivot_gc = df_pivot_gc[['HTGC', 'ATGC']]

        # Concatemamos (ordenando las columnas) y retornamos el nuevo DataFrame
        self.df = (pd.concat([df, df_pivot_gs, df_pivot_gc], axis=1)
                   .drop('year', axis=1)
                   )


def read_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['DateTime'])
    return df


def validate_features(df: pd.DataFrame):
    team = 'Melgar'
    print(df[(df['Home Team'] == team) | (df['Away Team'] == team)].head(10))


def classify(df: pd.DataFrame, num_discrete, num_continous, cat_cols):
    X = df.copy()
    y = X.pop('Result')

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=rs)

    preproc = ColumnTransformer([
        ('num_cont', StandardScaler(), num_continous),
        ('num_disc', StandardScaler(), num_discrete),
        ('cat', OneHotEncoder(handle_unknown='ignore'),
         cat_cols),
    ], remainder='drop')

    pipe = Pipeline([
        ('preproc', preproc),
        ('clf', LogisticRegression()),
    ])

    # TODO: Feature Selection Branch 1: MI Score
    # TODO: Feature Selection Branch 2: f_classif

    # TODO: Concatenación

    param_grid = [
        {
            'clf': [LogisticRegression()],
            'clf__random_state': [rs],
            'clf__solver': ['liblinear'],
            'clf__penalty': ['l1', 'l2']
        },
        {
            'clf': [SVC()],
            'clf__kernel': ['linear'],
            'clf__random_state': [rs],
        },
        {
            'clf': [DecisionTreeClassifier()],
            'clf__random_state': [rs],
            'clf__criterion': ['gini'],
        },
        {
            'clf': [RandomForestClassifier()],
            'clf__n_estimators': [20, 50, 100],
            'clf__random_state': [rs],
        },
        {
            'clf': [XGBClassifier()],
            'clf__learning_rate': [0.01, 0.1],
            'clf__objective': ['binary:logistic'],
            'clf__random_state': [rs],
        }
    ]

    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(X_train, y_train)

    y_pred = grid.best_estimator_.predict(X_test)

    print('Accuracy in validation:', accuracy_score(y_test, y_pred))
    print('Best Score in CV (Training):', grid.best_score_)
    print('Best estimator:',
          grid.best_estimator_.named_steps['clf'].__class__.__name__)


if __name__ == '__main__':
    # Random State
    rs = 69

    path = 'data/dataset/2020-2024 Matches Liga 1 Teams.csv'
    df = read_data(path)

    df = DataCleaning(df).clean_data()

    fe = FeatureEngineering(df)
    df = fe.baseline_fe()

    num_discrete = ['HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTGD', 'ATGD', 'MW']
    num_continuos = ['HTGSForm', 'ATGSForm', 'HTGCForm', 'ATGCForm']
    cat_cols = ['Round']

    print('Baseline:')
    classify(df, num_discrete, num_continuos, cat_cols)

    df = fe.augment_features()

    num_discrete += ['HTP', 'ATP', 'PD']
    print('\nAugmented:')
    classify(df, num_discrete, num_continuos, cat_cols)
