"""Cache CupGuard predictions in Redis."""

import json
import sys
from pathlib import Path

import redis

PART_B_SRC = Path(__file__).resolve().parents[1] / "Part-B" / "cupguard" / "src"
sys.path.insert(0, str(PART_B_SRC))

from pipeline import predict_fixture

r = redis.Redis(host="localhost", port=6379, decode_responses=True)


def cache_key(home: str, away: str, neutral: int = 1) -> str:
    return f"cupguard:pred:{home}:{away}:n{neutral}"


def get_prediction(home: str, away: str, neutral: int = 1) -> dict:
    key = cache_key(home, away, neutral)
    cached = r.get(key)
    if cached:
        print("cache HIT")
        return json.loads(cached)

    print("cache MISS")
    result = predict_fixture(home, away, neutral=neutral, is_world_cup=1)
    r.set(key, json.dumps(result))
    return result


if __name__ == "__main__":
    for label in ("first call", "second call"):
        probs = get_prediction("Brazil", "Argentina", neutral=1)
        print(f"{label}: {probs}")
