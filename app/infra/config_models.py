# ============================================================
# 1. PYDANTIC MODELS
# ============================================================
import os
import json
import logging
from enum import Enum
from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator


class LoanHistory(BaseModel):
    date: str
    loan_type: str
    bank: str
    status: str  # active | closed | default | restructured
    emi_pending: float = 0.0
    total_sanctioned: float = 0.0
    missed_payments: int = 0
 
 
class Dependent(BaseModel):
    name: str
    identifier: str
    income: float = 0.0
    has_loans: bool = False
    outstanding: float = 0.0
 
 
class RiskDecision(str, Enum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    REJECT = "REJECT"
 
 
class MLScores(BaseModel):
    pd_score: float = Field(ge=0.0, le=1.0, description="Probability of Default")
    interest_rate_forecast: float = Field(description="Predicted interest rate %")
    sector_risk_score: float = Field(ge=0.0, le=1.0)
    country_risk_score: float = Field(ge=0.0, le=1.0)
    feature_importances: dict = Field(default_factory=dict)
 
 
class CIBILDetails(BaseModel):
    score: int = Field(ge=300, le=900)
    total_accounts: int = 0
    overdue_accounts: int = 0
    credit_utilization: float = 0.0
    enquiries_last_6m: int = 0
 
 
class CreditRiskSummary(BaseModel):
    narrative: str
    decision: RiskDecision
    confidence: float = Field(ge=0.0, le=1.0)
    key_risk_drivers: list[str] = Field(min_length=1, max_length=10)
    mitigating_factors: list[str] = Field(default_factory=list)
    recommended_conditions: list[str] = Field(default_factory=list)
 
 
class UnderwritingState(BaseModel):
    """LangGraph state — single source of truth across all nodes."""
 
    # Input
    loan_account_id: str = ""
    identifier_person: str = ""
    applicant_name: str = ""
    loan_amount: float = 0.0
    loan_purpose: str = ""
    current_income: float = 0.0
    current_dti: float = 0.0
    employment_years: float = 0.0
    history: list[LoanHistory] = Field(default_factory=list)
    dependents: list[Dependent] = Field(default_factory=list)
 
    # Intermediate (populated by nodes)
    cibil_details: Optional[CIBILDetails] = None
    ml_scores: Optional[MLScores] = None
    risk_summary: Optional[CreditRiskSummary] = None
 
    # Control flow
    is_data_sufficient: bool = False
    clarification_questions: list[str] = Field(default_factory=list)
    clarification_count: int = 0
    max_clarifications: int = 2
    cache_hit: bool = False
    error_log: list[str] = Field(default_factory=list)
 
    # Metadata
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
 
    @field_validator("current_dti")
    @classmethod
    def validate_dti(cls, v):
        if v < 0:
            raise ValueError("DTI cannot be negative")
        return v