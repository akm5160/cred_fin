from typing import Optional

def cache_get(loan_id: str) -> Optional[dict]:
    """
    TODO: Implement Redis/Postgres cache lookup.
    - Key: loan_account_id
    - TTL: 1 hour
    - Returns cached {ml_scores, risk_summary} or None.
    """
    return None

def cache_set(loan_id: str, result: dict):
    """
    TODO: Implement Redis SETEX with TTL.
    - Stores ML scores + risk summary for loan_id.
    """
    pass