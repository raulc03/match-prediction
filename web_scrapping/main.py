from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
import random
from datetime import datetime

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.1901.188 Safari/537.36 Edg/115.0.1901.188",
    "Mozilla/5.0 (Linux; Android 13; SM-G990B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Android 13; Mobile; rv:120.0) Gecko/120.0 Firefox/120.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 OPR/100.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.5790.170 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.5790.170 Safari/537.36 Edg/115.0.5790.170",
]

base_url = 'https://fbref.com'

url_teams = []

def get_urls(urls: pd.DataFrame, years: list) -> pd.DataFrame:
    df = urls.copy()
    for year in years:
        # Liga 1 URL for each year
        url = f'{base_url}/en/comps/44/{year}/{year}-Liga-1-Stats'

        # Rotating headers to avoid block
        headers = {
            'User-Agent': random.choice(user_agents),
        }

        # URL request and parsing
        res = requests.get(url, headers=headers)

        if res.status_code == 200:
            soup = BeautifulSoup(res.content, 'html.parser')
            # Reviewing  the page, it was concluded that the first table
            # always has the list of teams with their URLs
            table = soup.find('table')
            for row in table.find('tbody').find_all('tr'): # type:ignore
                url = base_url + row.find('a').get('href') # type:ignore
                df.loc[len(df)] = [year, url]
        else:
            print(f'Failed to retrieve {url}')

        # Delay of each request to avoid blockage
        sp = random.randint(1, 3)
        time.sleep(sp)

    return df

def validate_years(path: str, years: list):
    urls_df = pd.DataFrame({'year': [], 'url': []})
    try:
        df = pd.read_csv(path)
        urls_df = df.copy()

        df_years = set(urls_df['year'].astype(int).unique().tolist())

        # Calculate missing years to obtain the URLs of the teams
        missing_years = sorted(list(set(years) - df_years))

        if len(missing_years) > 0:
            urls_df = get_urls(urls_df, missing_years)
            # Write the file with the missing URLs
            urls_df.to_csv(path, index=False)
        else:
            print(f'All URLs in the range {years[0]} - {years[-1]} have been written in: {path}')

    except (FileNotFoundError, pd.errors.EmptyDataError):
        print('Error with the file of URLs')

        # Gets all URLs in the year range
        urls_df = get_urls(urls_df, years)

        print('Creating the url file...')
        # Create the file with all the URLs
        urls_df.to_csv(path, index=False)

def get_stats(df: pd.DataFrame, path: str) -> None:
    url_df = pd.read_csv(path)

    print(f'Retrieving {url_df.shape[0]} URLs...')

    for url in url_df.loc[:, 'url']:
        headers = {
            'User-Agent': random.choice(user_agents),
        }

        res = requests.get(url, headers=headers)

        if res.status_code == 200:
            soup = BeautifulSoup(res.content, 'html.parser')

            team_from_url = url.split('/')[-1].split('Stats')[0]
            team = team_from_url.replace('-', ' ')

            for row in soup.find(id='matchlogs_for').find('tbody').find_all('tr'): # type:ignore
                date = [row.find('th').text.strip()] # type:ignore
                data = [d.text.strip() for d in row.find_all('td')] # type:ignore

                # Set new row
                table_data = date + data + [team]

                if table_data[0] != '':
                    df.loc[len(df)] = table_data
        else:
            print('Error with the URL:', url, res.status_code)

        sp = random.randint(4, 6)
        time.sleep(sp)

def main():
    initial_year: int = 2014
    last_year: int = datetime.now().year

    years = [year for year in range(initial_year, last_year)]

    # NOTE: Static route
    urls_path = 'data/url_teams.csv'
    validate_years(urls_path, years)

    cols = ['Date', 'Time', 'Comp', 'Round', 'Day',
            'Venue', 'Result', 'GF', 'GA', 'Opponent',
            'Poss', 'Attendance', 'Captain', 'Formation',
            'Opp Formation', 'Referee', 'Match Report', 'Notes', 'Team']
    df = pd.DataFrame(columns=cols) # type:ignore

    get_stats(df, urls_path)

    # NOTE: Static route
    final_path = f'data/Liga_1_Matches_{initial_year}-{last_year - 1}.csv'

    print(f'Saving file with statistics in:', final_path)
    df.to_csv(final_path, index=False)


# TODO: Obtener el año inicial por argumento
# TODO: Obtener el directorio de la data/ por argumento
if __name__ == '__main__':
    main()
