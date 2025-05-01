import time
import random
import pandas as pd
from bs4 import BeautifulSoup
import requests
import requests_cache
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
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

BASE_URL = 'https://fbref.com'

url_teams = []

def configure_session(
    cache_name: str = "fbref_cache",
    cache_expire: int = 3600
) -> requests.Session:
    # 1) Cache local
    requests_cache.install_cache(cache_name, expire_after=cache_expire)

    # 2) Sesión y cabeceras "browser-like"
    session = requests.Session()
    session.headers.update({
        "Accept-Language": "es-ES,es;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
        "Referer": BASE_URL + "/en",
        "Origin": BASE_URL,
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-User": "?1",
        "Sec-Fetch-Dest": "document",
        "Sec-Ch-Ua": '"Chromium";v="120", "Google Chrome";v="120"',
        "Sec-Ch-Ua-Mobile": "?0",
    })

    # 3) Retries con backoff exponencial
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def get_urls(urls: pd.DataFrame, years: list) -> pd.DataFrame:
    df = urls.copy()
    for year in years:
        # Liga 1 URL for each year
        url = f'{BASE_URL}/en/comps/44/{year}/{year}-Liga-1-Stats'

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
                url = BASE_URL + row.find('a').get('href') # type:ignore
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

def get_stats(
    session: requests.Session,
    df: pd.DataFrame,
    path: str
) -> None:
    url_df = pd.read_csv(path)
    total = url_df.shape[0]
    print(f"Retrieving {total} URLs...")

    for idx, url in enumerate(url_df['url'], start=1):
        session.headers['User-Agent'] = random.choice(user_agents)

        resp = session.get(url)
        status = resp.status_code

        if status == 200:
            soup = BeautifulSoup(resp.content, 'html.parser')
            team_slug = url.rstrip('/').split('/')[-1].split('Stats')[0]
            team = team_slug.replace('-', ' ').strip()

            table = soup.find(id='matchlogs_for')
            if table and table.tbody: # type:ignore
                for row in table.tbody.find_all('tr'): # type:ignore
                    date = row.find('th').get_text(strip=True) # type:ignore
                    data = [d.get_text(strip=True) for d in row.find_all('td')] # type:ignore
                    row_data = [date] + data + [team]

                    if row_data[0] != '':
                        if len(row_data) != len(df.columns):
                            print("Column mismatch:", url, len(row_data), len(df.columns))
                            # Ignore the xG and xGA columns.
                            del row_data[10:12]
                        df.loc[len(df)] = row_data

        elif status == 429:
            retry_after = int(resp.headers.get('Retry-After', 60))
            print(f"[{idx}/{total}] 429 on {url}, retry after {retry_after}s")
            time.sleep(retry_after + 1)
            continue
        else:
            print(f"[{idx}/{total}] Error {status} on {url}")

        # 4) Delay
        delay = max(6, random.gauss(8, 1.5))
        time.sleep(delay)

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
    df_stats = pd.DataFrame(columns=cols) # type:ignore

    sess = configure_session()

    get_stats(sess, df_stats, urls_path)

    # NOTE: Static route
    final_path = f'data/Liga_1_Matches_{initial_year}-{last_year - 1}.csv'

    print(f'Saving file with statistics in:', final_path)
    df_stats.to_csv(final_path, index=False)


if __name__ == '__main__':
    main()
