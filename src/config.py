# src/config.py
import os

# --- DATASET CONFIGURATION ---
DATA_FILENAME = "Data_DBH_50_hec.csv"

# Column Definitions
COL_ID = 'TAG' # Unique Tree Identifier
COL_SPECIES = 'SP' # Species Code
COL_SPECIES_GRP = 'Species Group' # Species Grouping
COL_X = 'XCO' # X Coordinate
COL_Y = 'YCO' # Y Coordinate

# Time-Step Logic
COL_CURRENT = 'D05'   # 2005
COL_HISTORY = 'D00'   # 2000
COL_TARGET  = 'D10'   # 2010

# Settings
DEFAULT_MIN_DBH = 5.0 # Minimum Diameter at Breast Height (cm)

# Model Artifacts
MODEL_FILENAME = "forest_growth_model.pkl"       # Regression (Size)
MORTALITY_MODEL_FILENAME = "forest_mortality_model.pkl" # Classification (Mortality)
ENCODER_FILENAME = "species_encoder.pkl" # Species One-Hot Encoder

# Parameters
DENSITY_RADIUS = 5.0 # in meters
COMPETITION_RADIUS = 6.0 # in meters
TEST_SIZE = 0.2 # Proportion of data for testing
RANDOM_STATE = 42 # For reproducibility
DEFAULT_GROWTH_PERCENTILE = 25 # Default percentile for growth threshold
DEFAULT_COMPETITION_INDEX = 0.5 # Default competition index threshold