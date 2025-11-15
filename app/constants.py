"""
Constants for IoT Threat Detection API

This module defines all magic numbers and configuration constants used throughout
the application, providing clear documentation and easy maintainability.
"""

# Feature expansion coefficients
# These values are used when converting simple 6-field input to 42 ML features

# Scaled rate coefficient - empirically determined from training data
SCALED_RATE_COEFFICIENT = 0.8

# Inter-arrival time minimum value (prevents division by zero)
IAT_MINIMUM = 0.001

# Packet size statistical multipliers
PACKET_SIZE_MIN_MULTIPLIER = 0.5  # Min packet size as fraction of average
PACKET_SIZE_MAX_MULTIPLIER = 1.5  # Max packet size as fraction of average
PACKET_SIZE_STD_MULTIPLIER = 0.3  # Standard deviation as fraction of average

# Statistical feature default values
# These are placeholder values when protocol information is unavailable
DEFAULT_RADIUS = 25.0
DEFAULT_COVARIANCE = 0.1
DEFAULT_VARIANCE = 0.2
DEFAULT_WEIGHT = 1.0
DEFAULT_PSH_FLAGS = 2.0

# Risk level thresholds
RISK_THRESHOLD_UNCERTAIN = 0.7  # Below this confidence = medium risk
RISK_THRESHOLD_LIKELY = 0.9     # Below this confidence = high risk
                                # Above this confidence = critical risk

# Memory management
MAX_RECENT_PREDICTIONS = 100  # Maximum predictions to store in memory
