from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
import os

def fetch_games(season="2024-25"):
    """
    Fetch game logs from the NBA API.
    Paste your existing nba_api logic in here.
    Saves to CSV for now — DB hookup comes later.
    """
    print(f"Fetching games for {season}...")
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
    games = gamefinder.get_data_frames()[0]
    os.makedirs("data/processed", exist_ok=True)
    games.to_csv("data/processed/games.csv", index=False)
    print(f"Saved {len(games)} games to data/processed/games.csv")
    return games

if __name__ == "__main__":
    fetch_games()