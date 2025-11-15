"""
Security utilities for IoT Threat Detection API

This module provides security features including:
- API key authentication
- Rate limiting
- Input validation with size limits
- Error message sanitization
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Dict

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

# API Key Authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Simple in-memory API key store (replace with database in production)
# To generate a key: python -c "import secrets; print(secrets.token_urlsafe(32))"
VALID_API_KEYS = {
    "dev-key-12345": {"name": "development", "rate_limit": "100/minute"},
    # Add more keys here or load from environment/database
}

# Input validation limits
MAX_NUMERIC_VALUE = 1e12  # Maximum value for numeric inputs
MIN_NUMERIC_VALUE = -1e12  # Minimum value for numeric inputs


def verify_api_key(api_key_header: str = Security(API_KEY_HEADER)) -> Dict[str, Any]:
    """
    Verify API key from request header.

    Args:
        api_key_header: API key from X-API-Key header

    Returns:
        Dictionary with API key metadata

    Raises:
        HTTPException: If API key is missing or invalid
    """
    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include X-API-Key header.",
        )

    if api_key_header not in VALID_API_KEYS:
        logger.warning("Invalid API key attempt: %s", api_key_header[:10] + "...")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return VALID_API_KEYS[api_key_header]


def validate_numeric_range(value: float, field_name: str) -> None:
    """
    Validate that a numeric value is within acceptable range.

    Args:
        value: Numeric value to validate
        field_name: Name of the field (for error messages)

    Raises:
        HTTPException: If value is out of acceptable range
    """
    if value > MAX_NUMERIC_VALUE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} exceeds maximum allowed value ({MAX_NUMERIC_VALUE})",
        )

    if value < MIN_NUMERIC_VALUE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} is below minimum allowed value ({MIN_NUMERIC_VALUE})",
        )


def sanitize_error_message(error: Exception, show_details: bool = False) -> str:
    """
    Sanitize error messages to prevent information disclosure.

    Args:
        error: The exception that occurred
        show_details: Whether to show detailed error messages (dev mode only)

    Returns:
        Sanitized error message safe for public display
    """
    if show_details:
        return str(error)

    # Map specific errors to generic messages
    error_type = type(error).__name__

    generic_messages = {
        "ValueError": "Invalid input data",
        "TypeError": "Invalid data type",
        "KeyError": "Missing required field",
        "FileNotFoundError": "Resource not found",
        "PermissionError": "Access denied",
        "ConnectionError": "Service temporarily unavailable",
    }

    return generic_messages.get(error_type, "An internal error occurred")


# CORS configuration
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:5000",
    # Add production domains here
]

CORS_CONFIG = {
    "allow_origins": CORS_ORIGINS,
    "allow_credentials": True,
    "allow_methods": ["GET", "POST"],
    "allow_headers": ["*"],
}
