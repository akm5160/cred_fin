def check_rate_limit(user_id: str) -> tuple[bool, str]:
    """
    TODO: Implement Redis-backed sliding window rate limiter.
    - Per-user request quota (e.g., 60 req/min)
    - App-level daily token budget (e.g., 500K tokens/day)
    Returns (allowed: bool, reason: str).
    """
    return True, "OK"