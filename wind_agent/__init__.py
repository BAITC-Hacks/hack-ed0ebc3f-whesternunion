"""Wind generation forecasting with auditable data cutoffs."""
import os

# Portable single-worker default, including Windows without the legacy WMIC tool.
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')
