import pandas as pd
from pathlib import Path
import re


def data_cleaning(df: pd.DataFrame, team: str):
    # Eliminamos columnas innecesarias
    df.drop(['Unnamed: 0', 'Match Report', 'Notes'], inplace=True, axis=1)

    # Agregamos el equipo al que el dataset hace referencia
    df['Team'] = team

    # Parsear la fecha
    fecha_hora = df['Date'] + ' ' + df['Time']
    df['DateTime'] = pd.to_datetime(fecha_hora, format='%Y-%m-%d %H:%M')
    df['Day'] = df['DateTime'].dt.dayofweek

    # Limpiar NaN
    total_rows = df.shape[0]
    drop_columns = []
    for col in df.columns:
        nan = df[col].isnull().sum()
        por = nan / total_rows
        if (por > 0.7):
            drop_columns.append(col)

    df.drop(drop_columns, axis=1, inplace=True)

    return df


if __name__ == '__main__':
    data_dir = './data/'
    files_ext = '*.csv'

    files_path = Path(data_dir).glob(files_ext)
    # columns = ['Date', 'Time', 'Comp', 'Round', 'Day', 'Venue', 'Result',
    #            'GF', 'GA', 'Opponent', 'Captain', 'Formation', 'Opp Formation',
    #            'Referee', 'Team', 'DateTime']
    df_final = None

    cnt = 0
    for file in files_path:
        team = ''
        path_file = str(file.relative_to(data_dir))
        match = re.search(r'(.+?) Stats', path_file)
        if match:
            team = match.group(1)
            df = data_cleaning(pd.read_csv(file), team)
            if type(df_final) == pd.DataFrame:
                cnt += 1
                df_final = pd.concat([df_final, df], ignore_index=True)
            else:
                print('Test')
                df_final = df
        else:
            print('Error, no se obtuvo el equipo')
            break

    if type(df_final) == pd.DataFrame:
        df_final.to_csv(
            data_dir + 'dataset/2020-2024 Matches Liga 1 Teams.csv')

    print(cnt)
