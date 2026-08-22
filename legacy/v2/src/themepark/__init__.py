"""Theme park wait-time forecasting: shared library for ingestion, training and serving.

Every entry point -- local scripts, GitHub Actions jobs and Databricks tasks -- imports
from this package. Nothing reimplements feature engineering or name sanitisation.
"""

__version__ = "2.0.0"
