# src/config.py
import os

# --- DATASET CONFIGURATION ---
DATA_FILENAME = "Data_DBH_50_hec.csv" # Path to the main dataset CSV file

# Column Definitions (Single Source of Truth)
COL_ID = 'TAG' # Unique Tree Identifier
COL_SPECIES = 'SP' # Species Code
COL_SPECIES_GRP = 'Species Group' # Species Grouping
COL_DBH_PREFIX = 'D' # Prefix for DBH columns (e.g., D00, D05, D10, etc.)
COL_X = 'XCO' # X Coordinate
COL_Y = 'YCO'  # Y Coordinate

# Time-Step Logic:
# We treat 2005 (D05) as the "Current" state for the dashboard.
COL_CURRENT = 'D05'   # 2005
COL_HISTORY = 'D00'   # 2000 (Used to calculate past growth)
COL_TARGET  = 'D10'   # 2010 (Used as the training target)

# Model Artifacts
MODEL_FILENAME = "forest_growth_model.pkl"
ENCODER_FILENAME = "species_encoder.pkl"

# Parameters
DENSITY_RADIUS = 5.0
COMPETITION_RADIUS = 6.0
TEST_SIZE = 0.2
RANDOM_STATE = 42
DEFAULT_GROWTH_PERCENTILE = 25
DEFAULT_PROXIMITY = 5.0