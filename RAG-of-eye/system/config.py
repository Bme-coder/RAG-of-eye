"""
Global configuration values for the Myopia Mining & Analytics pipeline.

These thresholds are referenced by the upcoming saturated mining workflow to
ensure that extracted records fall within medically reasonable bounds before
being aggregated.
"""

# Annual SER progression limits (diopters / year)
MAX_PROGRESSION_RATE = -2.5
MIN_PROGRESSION_RATE = 0.5

# Minimum cohort size required for a record to participate in statistics
MIN_SAMPLE_SIZE = 10

# Default standard deviations used when no study variance is available
DEFAULT_SD_NATURAL = 0.25
DEFAULT_SD_TREATMENT = 0.20
