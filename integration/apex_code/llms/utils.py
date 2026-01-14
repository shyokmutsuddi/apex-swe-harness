"""Utility functions for LLM operations."""

from ..config import (
    DEFAULT_TEMPERATURE,
    MODELS_NOT_SUPPORTING_TEMP,
    REQUIRED_TEMPERATURE_1_0,
)


def get_model_temperature(
    model_name: str, user_temperature: float | None = None
) -> float:
    """
    Get the appropriate temperature for a given model.

    Args:
        model_name: The name of the model
        user_temperature: User-specified temperature (if any)

    Returns:
        The temperature to use for this model
    """
    if user_temperature is not None:
        return user_temperature

    # Check if model doesn't support temperature control
    if model_name in MODELS_NOT_SUPPORTING_TEMP:
        return REQUIRED_TEMPERATURE_1_0

    return DEFAULT_TEMPERATURE
