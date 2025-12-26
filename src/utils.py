# src/utils.py
import pandas as pd
import joblib
from scipy.spatial import cKDTree
import numpy as np
import streamlit as st
from .config import * # Imports COL_CURRENT, COL_HISTORY, etc.

@st.cache_resource
def load_model_resources():
    try:
        model = joblib.load(MODEL_FILENAME)
        encoder = joblib.load(ENCODER_FILENAME)
        return model, encoder
    except FileNotFoundError:
        return None, None

def calculate_competition_index(df, radius=COMPETITION_RADIUS):
    # Dynamic Check: Do we have the "Current" column?
    if COL_CURRENT not in df.columns: return df

    # Drop invalid rows for spatial calc
    valid_trees = df.dropna(subset=[COL_X, COL_Y, COL_CURRENT]).copy().reset_index(drop=True)
    coords = valid_trees[[COL_X, COL_Y]].values
    dbh = valid_trees[COL_CURRENT].values
    
    tree = cKDTree(coords)
    neighbors_list = tree.query_ball_point(coords, r=radius)
    
    ci_scores = []
    for i, neighbors in enumerate(neighbors_list):
        subject_dbh = dbh[i]
        subject_ci = 0
        for n_idx in neighbors:
            if i == n_idx: continue
            dist = np.linalg.norm(coords[i] - coords[n_idx])
            if dist > 0:
                n_dbh = dbh[n_idx]
                # Only calculate if both trees are alive (dbh > 0)
                if n_dbh > 0 and subject_dbh > 0:
                    subject_ci += (n_dbh / subject_dbh) / dist
        ci_scores.append(subject_ci)
        
    valid_trees['Competition_Index'] = ci_scores
    return df.merge(valid_trees[[COL_ID, 'Competition_Index']], on=COL_ID, how='left')

@st.cache_data
def load_and_process_data(csv_path):
    try:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
        except:
            df = pd.read_csv(csv_path, encoding='latin1')

        df.columns = df.columns.str.strip()
        
        # Clean Text Columns
        if COL_SPECIES in df.columns: 
            df[COL_SPECIES] = df[COL_SPECIES].astype(str).str.strip()
        if COL_SPECIES_GRP in df.columns: 
            df[COL_SPECIES_GRP] = df[COL_SPECIES_GRP].astype(str).str.strip()
        
        # Clean Numeric Columns
        cols_to_clean = [COL_HISTORY, COL_CURRENT, COL_TARGET]
        for c in cols_to_clean:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                df[c].replace(0, np.nan, inplace=True)

        # 1. Feature: Past Growth (History -> Current)
        if COL_HISTORY in df.columns and COL_CURRENT in df.columns:
            df['GROWTH_HIST'] = df[COL_CURRENT] - df[COL_HISTORY]

        # 2. Spatial Calculations (Based on Current State)
        spatial_df = df.dropna(subset=[COL_X, COL_Y, COL_CURRENT]).copy()
        if not spatial_df.empty:
            coords = spatial_df[[COL_X, COL_Y]].values
            tree = cKDTree(coords)
            
            # Distance
            dists, _ = tree.query(coords, k=2)
            spatial_df['Nearest_Neighbor_Dist'] = dists[:, 1]
            
            # Density
            counts = tree.query_ball_point(coords, r=DENSITY_RADIUS, return_length=True)
            spatial_df['Local_Density'] = counts - 1
            
            df = df.merge(spatial_df[[COL_ID, 'Nearest_Neighbor_Dist', 'Local_Density']], on=COL_ID, how='left')

        df = calculate_competition_index(df)

        # Drop rows that lack data for App visualization
        df.dropna(subset=[COL_CURRENT, 'GROWTH_HIST', 'Competition_Index'], inplace=True)
        
        return df

    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return None

def run_predictions(df, model, encoder):
    # Encode Species
    if 'SP_Encoded' not in df.columns:
        known_classes = encoder.classes_
        map_dict = {sp: i for i, sp in enumerate(known_classes)}
        df['SP_Encoded'] = df[COL_SPECIES].map(map_dict).fillna(-1) 
    
    # Feature Vector (Using Config Constants)
    # The model will be trained on these exact column names
    features = [COL_CURRENT, 'GROWTH_HIST', 'Nearest_Neighbor_Dist', 'Local_Density', 'Competition_Index', 'SP_Encoded']
    
    pred_df = df[features].dropna()
    pred_df = pred_df[pred_df['SP_Encoded'] != -1]
    
    if not pred_df.empty:
        # Predict Future Size (e.g., D10)
        predicted_future = model.predict(pred_df)
        
        df.loc[pred_df.index, 'Predicted_Size'] = predicted_future 
        df.loc[pred_df.index, 'Predicted_Growth'] = predicted_future - pred_df[COL_CURRENT]
        
    return df