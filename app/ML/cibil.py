
from app.infra.config_models import CIBILDetails
def fetch_cibil(person_id: str) -> CIBILDetails:
    """
    TODO: Implement real CIBIL/Experian API call.
    - HTTP POST to credit bureau endpoint
    - Retry with exponential backoff (3 attempts)
    - Returns CIBILDetails or raises on failure.
    """
    return CIBILDetails(
        score=680, total_accounts=5, overdue_accounts=0,
        credit_utilization=45.0, enquiries_last_6m=2,
    )
