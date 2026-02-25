import json
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np

# Defaults (can be overridden via constructor)
DEFAULT_NUM_PARTICLES = 20000
DEFAULT_PRIOR_BETA = 2  # Beta(2,2) prior for captain call probability
DEFAULT_SIGMA_PRIOR_COIN = 0.12
DEFAULT_SIGMA_PRIOR_CONTEXT = 0.10  # venue/captain priors
DEFAULT_SIGMA_COIN_DRIFT = 0.015
DEFAULT_SIGMA_CONTEXT_DRIFT = 0.0   # small rejuvenation to avoid impoverishment (0 disables)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class TossParticleFilter:
    """
    Particle filter for modeling cricket toss outcomes with latent biases:
    - Global coin bias
    - Venue-specific bias
    - Captain-specific bias (for the captain who calls)

    The model updates per observed toss and can predict the probability that
    a given captain (if they call) wins the toss at a venue. It can also
    estimate team-level toss win probabilities for two captains when the caller
    is unknown.
    """

    def __init__(
        self,
        num_particles: int = DEFAULT_NUM_PARTICLES,
        prior_beta: int = DEFAULT_PRIOR_BETA,
        sigma_prior_coin: float = DEFAULT_SIGMA_PRIOR_COIN,
        sigma_prior_context: float = DEFAULT_SIGMA_PRIOR_CONTEXT,
        sigma_coin_drift: float = DEFAULT_SIGMA_COIN_DRIFT,
        sigma_context_drift: float = DEFAULT_SIGMA_CONTEXT_DRIFT,
    ) -> None:
        self.N = int(num_particles)
        self.prior_beta = int(prior_beta)
        self.sigma_prior_coin = float(sigma_prior_coin)
        self.sigma_prior_context = float(sigma_prior_context)
        self.sigma_coin_drift = float(sigma_coin_drift)
        self.sigma_context_drift = float(sigma_context_drift)

        # Latent states
        self.coin_bias = np.random.normal(0.0, self.sigma_prior_coin, self.N)
        self.venue_bias: Dict[str, np.ndarray] = {}
        self.location_bias: Dict[str, np.ndarray] = {}
        self.captain_bias: Dict[str, np.ndarray] = {}

        # Captain calling behavior counts
        self.captain_heads = defaultdict(int)
        self.captain_total = defaultdict(int)

        # Optional last ingested date string for external time-scaling
        self.last_date: Optional[str] = None

    def _get_bias_array(self, key: str, store: Dict[str, np.ndarray]) -> np.ndarray:
        if key not in store:
            store[key] = np.random.normal(0.0, self.sigma_prior_context, self.N)
        return store[key]

    def captain_call_prob(self, captain: str) -> float:
        h = self.captain_heads[captain]
        t = self.captain_total[captain]
        prior = self.prior_beta
        return (h + prior) / (t + 2 * prior)

    def predict_step(self, step_scale: float = 1.0) -> None:
        # Global coin drift
        if self.sigma_coin_drift > 0:
            sigma = max(0.0, float(step_scale)) * self.sigma_coin_drift
            self.coin_bias += np.random.normal(0.0, sigma, self.N)

        # Optional slight rejuvenation for context-specific biases to avoid particle impoverishment
        if self.sigma_context_drift > 0:
            sigma_c = max(0.0, float(step_scale)) * self.sigma_context_drift
            for k, v in list(self.venue_bias.items()):
                self.venue_bias[k] = v + np.random.normal(0.0, sigma_c, self.N)
            for k, v in list(self.location_bias.items()):
                self.location_bias[k] = v + np.random.normal(0.0, sigma_c, self.N)
            for k, v in list(self.captain_bias.items()):
                self.captain_bias[k] = v + np.random.normal(0.0, sigma_c, self.N)

    def _p_heads_given_caller(self, venue: str, captain: str, location: Optional[str] = None) -> np.ndarray:
        venue_b = self._get_bias_array(venue, self.venue_bias)
        captain_b = self._get_bias_array(captain, self.captain_bias)
        if location:
            location_b = self._get_bias_array(location, self.location_bias)
        else:
            location_b = 0.0
        return sigmoid(self.coin_bias + venue_b + captain_b + location_b)

    def update(self, venue: str, captain: str, call: str, result: str, location: Optional[str] = None) -> None:
        # Normalize inputs
        call = call.strip().upper()
        result = result.strip().upper()
        if call not in {"H", "T"}:
            raise ValueError("call must be 'H' or 'T'")
        if result not in {"H", "T"}:
            raise ValueError("result must be 'H' or 'T'")

        # Update captain calling behavior counts
        self.captain_total[captain] += 1
        if call == "H":
            self.captain_heads[captain] += 1

        p_heads = self._p_heads_given_caller(venue, captain, location=location)
        outcome_heads = (result == "H")
        likelihood = p_heads if outcome_heads else (1.0 - p_heads)

        # Normalize weights robustly
        w = likelihood
        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            # fallback to uniform weights to avoid failure
            weights = np.full(self.N, 1.0 / self.N)
        else:
            weights = w / w_sum

        idx = np.random.choice(self.N, self.N, p=weights)

        # Resample relevant latent states
        self.coin_bias = self.coin_bias[idx]
        vb = self._get_bias_array(venue, self.venue_bias)
        cb = self._get_bias_array(captain, self.captain_bias)
        self.venue_bias[venue] = vb[idx]
        self.captain_bias[captain] = cb[idx]
        if location:
            lb = self._get_bias_array(location, self.location_bias)
            self.location_bias[location] = lb[idx]

    def predict(self, venue: str, captain: str, location: Optional[str] = None) -> Tuple[float, float, float]:
        """
        Predict toss win probability for the provided captain IF THEY CALL.

        Returns: (win_prob, win_std, call_prob)
        - win_std is the std of the final win probability (not p_heads)
        """
        p_heads = self._p_heads_given_caller(venue, captain, location=location)
        p_heads_mean = float(np.mean(p_heads))
        p_heads_std = float(np.std(p_heads))

        call_prob = self.captain_call_prob(captain)
        win_prob = call_prob * p_heads_mean + (1.0 - call_prob) * (1.0 - p_heads_mean)

        # std of affine transform (approx): |2*call_prob - 1| * std(p_heads)
        win_std = abs(2.0 * call_prob - 1.0) * p_heads_std

        return float(win_prob), float(win_std), float(call_prob)

    def predict_two_captains(
        self,
        venue: str,
        captain_a: str,
        captain_b: str,
        caller: Optional[str] = None,
        location: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Estimate team-level toss win probabilities for two captains.

        If caller is provided ('A', 'B', captain name matching either, or team labels used as keys),
        the returned probabilities correspond to that caller scenario.
        If caller is None, assumes 50/50 chance either captain calls and mixes accordingly.
        Returns dict with keys: 'A_calls_A_wins', 'B_calls_B_wins', 'A_overall', 'B_overall'.
        """
        # Scenario: A calls
        p_heads_A = self._p_heads_given_caller(venue, captain_a, location=location)
        call_A = self.captain_call_prob(captain_a)
        pA_wins_if_A_calls = call_A * float(np.mean(p_heads_A)) + (1.0 - call_A) * (1.0 - float(np.mean(p_heads_A)))

        # Scenario: B calls
        p_heads_B = self._p_heads_given_caller(venue, captain_b, location=location)
        call_B = self.captain_call_prob(captain_b)
        pB_wins_if_B_calls = call_B * float(np.mean(p_heads_B)) + (1.0 - call_B) * (1.0 - float(np.mean(p_heads_B)))

        # If caller is known
        caller_norm = (caller or "").strip()
        if caller_norm:
            if caller_norm.upper() == "A" or caller_norm == captain_a:
                return {
                    "A_calls_A_wins": float(pA_wins_if_A_calls),
                    "B_calls_B_wins": float(1.0 - pA_wins_if_A_calls),
                    "A_overall": float(pA_wins_if_A_calls),
                    "B_overall": float(1.0 - pA_wins_if_A_calls),
                }
            if caller_norm.upper() == "B" or caller_norm == captain_b:
                return {
                    "A_calls_A_wins": float(1.0 - pB_wins_if_B_calls),
                    "B_calls_B_wins": float(pB_wins_if_B_calls),
                    "A_overall": float(1.0 - pB_wins_if_B_calls),
                    "B_overall": float(pB_wins_if_B_calls),
                }

        # Unknown caller: assume 50/50
        A_overall = 0.5 * pA_wins_if_A_calls + 0.5 * (1.0 - pB_wins_if_B_calls)
        B_overall = 1.0 - A_overall
        return {
            "A_calls_A_wins": float(pA_wins_if_A_calls),
            "B_calls_B_wins": float(pB_wins_if_B_calls),
            "A_overall": float(A_overall),
            "B_overall": float(B_overall),
        }

    def save(self, filename: str = "data.json") -> None:
        data = {
            "version": 1,
            "config": {
                "N": self.N,
                "prior_beta": self.prior_beta,
                "sigma_prior_coin": self.sigma_prior_coin,
                "sigma_prior_context": self.sigma_prior_context,
                "sigma_coin_drift": self.sigma_coin_drift,
                "sigma_context_drift": self.sigma_context_drift,
            },
            "coin_bias": self.coin_bias.tolist(),
            "venue_bias": {k: v.tolist() for k, v in self.venue_bias.items()},
            "location_bias": {k: v.tolist() for k, v in self.location_bias.items()},
            "captain_bias": {k: v.tolist() for k, v in self.captain_bias.items()},
            "captain_heads": dict(self.captain_heads),
            "captain_total": dict(self.captain_total),
            "last_date": self.last_date,
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, filename: str = "data.json") -> None:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Backward compatibility with old schema (no version)
            if isinstance(data, dict) and "coin_bias" in data and "version" not in data:
                self.coin_bias = np.array(data["coin_bias"])  # type: ignore[index]
                self.N = int(self.coin_bias.shape[0])
                self.venue_bias = {k: np.array(v) for k, v in data.get("venue_bias", {}).items()}
                self.location_bias = {k: np.array(v) for k, v in data.get("location_bias", {}).items()}
                self.captain_bias = {k: np.array(v) for k, v in data.get("captain_bias", {}).items()}
                self.captain_heads = defaultdict(int, data.get("captain_heads", {}))
                self.captain_total = defaultdict(int, data.get("captain_total", {}))
                self.last_date = data.get("last_date")
                return

            # New schema
            cfg = data.get("config", {})
            coin = np.array(data["coin_bias"])  # type: ignore[index]
            self.N = int(coin.shape[0])
            self.coin_bias = coin
            self.venue_bias = {k: np.array(v) for k, v in data.get("venue_bias", {}).items()}
            self.location_bias = {k: np.array(v) for k, v in data.get("location_bias", {}).items()}
            self.captain_bias = {k: np.array(v) for k, v in data.get("captain_bias", {}).items()}
            self.captain_heads = defaultdict(int, data.get("captain_heads", {}))
            self.captain_total = defaultdict(int, data.get("captain_total", {}))
            self.last_date = data.get("last_date")

            # Update runtime config from file if present
            self.prior_beta = int(cfg.get("prior_beta", self.prior_beta))
            self.sigma_prior_coin = float(cfg.get("sigma_prior_coin", self.sigma_prior_coin))
            self.sigma_prior_context = float(cfg.get("sigma_prior_context", self.sigma_prior_context))
            self.sigma_coin_drift = float(cfg.get("sigma_coin_drift", self.sigma_coin_drift))
            self.sigma_context_drift = float(cfg.get("sigma_context_drift", self.sigma_context_drift))
        except FileNotFoundError:
            # Start fresh if no data file exists
            return
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to decode JSON from {filename}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load model state from {filename}: {e}") from e


class TossWinnerOnlyPF:
    """
    Particle filter that models toss winner directly without needing call/result.
    Features:
      - Global coin bias
      - Venue bias (ground)
      - Location bias (city/country)
      - Team bias (team-specific tendency)

    For a fixture (team_a vs team_b), the per-particle probability that A wins is:
        pA = sigmoid(coin + venue + location + team[a] - team[b])
    """

    def __init__(
        self,
        num_particles: int = DEFAULT_NUM_PARTICLES,
        sigma_prior_coin: float = DEFAULT_SIGMA_PRIOR_COIN,
        sigma_prior_context: float = DEFAULT_SIGMA_PRIOR_CONTEXT,
        sigma_coin_drift: float = DEFAULT_SIGMA_COIN_DRIFT,
        sigma_context_drift: float = DEFAULT_SIGMA_CONTEXT_DRIFT,
    ) -> None:
        self.N = int(num_particles)
        self.sigma_prior_coin = float(sigma_prior_coin)
        self.sigma_prior_context = float(sigma_prior_context)
        self.sigma_coin_drift = float(sigma_coin_drift)
        self.sigma_context_drift = float(sigma_context_drift)

        self.coin_bias = np.random.normal(0.0, self.sigma_prior_coin, self.N)
        self.venue_bias: Dict[str, np.ndarray] = {}
        self.location_bias: Dict[str, np.ndarray] = {}
        self.team_bias: Dict[str, np.ndarray] = {}

        self.last_date: Optional[str] = None

    def _get_bias_array(self, key: str, store: Dict[str, np.ndarray]) -> np.ndarray:
        if key not in store:
            store[key] = np.random.normal(0.0, self.sigma_prior_context, self.N)
        return store[key]

    def predict_step(self, step_scale: float = 1.0) -> None:
        if self.sigma_coin_drift > 0:
            sigma = max(0.0, float(step_scale)) * self.sigma_coin_drift
            self.coin_bias += np.random.normal(0.0, sigma, self.N)
        if self.sigma_context_drift > 0:
            sigma_c = max(0.0, float(step_scale)) * self.sigma_context_drift
            for k, v in list(self.venue_bias.items()):
                self.venue_bias[k] = v + np.random.normal(0.0, sigma_c, self.N)
            for k, v in list(self.location_bias.items()):
                self.location_bias[k] = v + np.random.normal(0.0, sigma_c, self.N)
            for k, v in list(self.team_bias.items()):
                self.team_bias[k] = v + np.random.normal(0.0, sigma_c, self.N)

    def _pA(self, venue: str, team_a: str, team_b: str, location: Optional[str] = None) -> np.ndarray:
        vb = self._get_bias_array(venue, self.venue_bias)
        lb = self._get_bias_array(location, self.location_bias) if location else 0.0
        ta = self._get_bias_array(team_a, self.team_bias)
        tb = self._get_bias_array(team_b, self.team_bias)
        return sigmoid(self.coin_bias + vb + lb + (ta - tb))

    def update(self, venue: str, team_a: str, team_b: str, winner: str, location: Optional[str] = None) -> None:
        pA = self._pA(venue, team_a, team_b, location=location)
        outcome_A = (winner == team_a)
        likelihood = pA if outcome_A else (1.0 - pA)

        w_sum = float(np.sum(likelihood))
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            weights = np.full(self.N, 1.0 / self.N)
        else:
            weights = likelihood / w_sum

        idx = np.random.choice(self.N, self.N, p=weights)
        self.coin_bias = self.coin_bias[idx]
        vb = self._get_bias_array(venue, self.venue_bias)
        self.venue_bias[venue] = vb[idx]
        if location:
            lb = self._get_bias_array(location, self.location_bias)
            self.location_bias[location] = lb[idx]
        ta = self._get_bias_array(team_a, self.team_bias)
        tb = self._get_bias_array(team_b, self.team_bias)
        self.team_bias[team_a] = ta[idx]
        self.team_bias[team_b] = tb[idx]

    def predict_two(self, venue: str, team_a: str, team_b: str, location: Optional[str] = None) -> Tuple[float, float]:
        pA = self._pA(venue, team_a, team_b, location=location)
        return float(np.mean(pA)), float(np.std(pA))

    def save(self, filename: str = "winner_data.json") -> None:
        data = {
            "version": 1,
            "config": {
                "N": self.N,
                "sigma_prior_coin": self.sigma_prior_coin,
                "sigma_prior_context": self.sigma_prior_context,
                "sigma_coin_drift": self.sigma_coin_drift,
                "sigma_context_drift": self.sigma_context_drift,
            },
            "coin_bias": self.coin_bias.tolist(),
            "venue_bias": {k: v.tolist() for k, v in self.venue_bias.items()},
            "location_bias": {k: v.tolist() for k, v in self.location_bias.items()},
            "team_bias": {k: v.tolist() for k, v in self.team_bias.items()},
            "last_date": self.last_date,
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, filename: str = "winner_data.json") -> None:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            coin = np.array(data["coin_bias"])  # type: ignore[index]
            self.N = int(coin.shape[0])
            self.coin_bias = coin
            self.venue_bias = {k: np.array(v) for k, v in data.get("venue_bias", {}).items()}
            self.location_bias = {k: np.array(v) for k, v in data.get("location_bias", {}).items()}
            self.team_bias = {k: np.array(v) for k, v in data.get("team_bias", {}).items()}
            cfg = data.get("config", {})
            self.sigma_prior_coin = float(cfg.get("sigma_prior_coin", self.sigma_prior_coin))
            self.sigma_prior_context = float(cfg.get("sigma_prior_context", self.sigma_prior_context))
            self.sigma_coin_drift = float(cfg.get("sigma_coin_drift", self.sigma_coin_drift))
            self.sigma_context_drift = float(cfg.get("sigma_context_drift", self.sigma_context_drift))
            self.last_date = data.get("last_date")
        except FileNotFoundError:
            return
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to decode JSON: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load winner-only model: {e}") from e
