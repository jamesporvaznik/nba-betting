import sys
import json

def predict(game_id):
    """
    Called by Express to get a prediction for a game.
    Must print valid JSON to stdout — don't print anything else.
    """
    result = {
        "game_id": game_id,
        "home_win_prob": 0.0,
        "status": "model not trained yet"
    }
    print(json.dumps(result))

if __name__ == "__main__":
    game_id = sys.argv[1] if len(sys.argv) > 1 else "1"
    predict(game_id)