Full-stack machine learning project predicting NBA player props.

Stack
- Data: nba_api, basketball reference, nbainjuries(PyPi)
- Datbase: Supabase
- Backend: Node.js / Express
- Frontend: Next.js / TypeScript / Tailwind CSS
- Data Storage / Handling: Python

Install dependencies
- cd api && npm install && cd ../frontend && npm install
- cd ../python && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

Run on localhost
- npm run dev

Data Collection and Datbase:

- Database tables:
  - players : collect basic player identification data as well as injury status.
    - insert_player_teams.py get all players and teams for a given season. All categories filled for players other than bball_ref_id. Use       update_bball_ids.py and Player_Shooting.csv to collect those ids.
  - teams : collect basic team identification data
    - insert_player_teams.py fills all this data.
  - player_gamelog : collects all simple box score stats of every player from every game in a given season.
    - Use player_season_gamelog.py to collect this data.


- Data I still need to add:
  - Location based shooting data for players from bball reference, although this is season based and not game by game.
  - Location based shooting data for defensive teams to see where they give up their points.
  - Team defensive stats
  - Use gamelog data to make distinctions such as days of rest, and home or away
  - Injury data from nbainjuries, look for how much usage increases when a certain player gets injured
  - Try and make comparisons based on positional and similar production in a given stat to see if for example if devin booker and donovan mitchell are both high scoring guards, so if devin booker does score a lot against okc maybe donovan mitchell wont either and look at how big of a trend it is. Iffy, game plan matters more probably, but there are some defenders like stephon castle that I think can lock up any guard.
  - If possible try and find stats for what play type a player scores from, and then see how well a team defends against those play types.
  - These are mostly stats for points so maybe look more for other data to predict other stats.
  
  - Pace and possessions : More possessions = more opportunity to get stats
  - Vegas lines : Need those so know what to compare predictions against
  - Minutes projection : More minutes = more opportunity
  - Opponent defensive matchup at the position level : positional defense can be more accurate than full team ranks
  - Recency weighting : maybe weigh recent performances more than season data
  - Game total (over/under) : more or less scoring opportunities based on the total


Practical Workflow for testing features:
1. Train baseline model with core features
2. Look at SHAP values → find which features the model leans on most
3. Form a basketball-logic hypothesis about an interaction
4. Test that specific combination
5. Did it improve calibration? Keep it. If not, discard it.
6. Repeat
