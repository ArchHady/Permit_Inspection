"""
Building Permits Analysis Dashboard
Interactive Streamlit application for San Francisco Building Permits data analysis
"""

# Standard library
import io
import time
import warnings
import pickle
import re
from datetime import datetime
from functools import wraps
from typing import Optional, Tuple, List, Dict, Any

# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Streamlit
import streamlit as st

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score

# Imbalanced data handling
from imblearn.over_sampling import SMOTE

from streamlit.runtime.scriptrunner import get_script_run_ctx

# =============================================================================
# CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="SF Building Permits Analysis",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .header-image img {
        width: 100%;
        height: 300px;
        object-fit: cover;
        object-position: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-image">', unsafe_allow_html=True)
st.image("https://images.contentstack.io/v3/assets/blt06f605a34f1194ff/bltc85bbcd6ff5fa0fd/650882a0a39cd61ce6ace86f/0_-_BCC-2023-THINGS-TO-DO-IN-SAN-FRANCISCO-AT-NIGHT-0.webp?fit=crop&disable=upscale&auto=webp&quality=60&crop=smart")
st.markdown('</div>', unsafe_allow_html=True)
class Config:
    """Application configuration"""
    RANDOM_STATE = 42
    CACHE_TTL = 3600  # 1 hour
    CHUNK_SIZE = 10000
    
    INACTIVE_STATUSES = [
        'complete', 'cancelled', 'expired', 
        'withdrawn', 'revoked', 'disapproved'
    ]
    
    IRRELEVANT_COLS = [
        'Permit Number', 'Record ID', 'Block', 'Lot',
        'Street Number', 'Street Number Suffix', 'Street Name',
        'Street Suffix', 'Unit', 'Unit Suffix', 'Description',
        'Location', 'Permit Creation Date', 'Current Status Date',
        'Completed Date', 'First Construction Document Date',
        'Structural Notification', 'Fire Only Permit',
        'Site Permit', 'Permit Type'
    ]
    
    DATE_COLUMNS = [
        'Permit Creation Date', 'Current Status Date', 'Filed Date',
        'Issued Date', 'Completed Date', 'First Construction Document Date',
        'Permit Expiration Date'
    ]
    
    # Extended valid statuses based on SF permit data
    VALID_STATUSES = [
        'issued', 'complete', 'filed', 'expired', 'withdrawn', 
        'cancelled', 'suspend', 'plancheck', 'reinstated',
        'approved', 'appeal', 'revoked', 'disapproved', 'incomplete'
    ]
    
    REQUIRED_COLUMNS = ['Current Status', 'Filed Date']

# Settings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
np.random.seed(Config.RANDOM_STATE)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        
        # Check if we're in a Streamlit context
        if get_script_run_ctx() is not None:
            if st.session_state.get('debug_mode', False):
                st.sidebar.text(f"{func.__name__}: {end-start:.2f}s")
        
        return result
    return wrapper

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'df_original': None,
        'df_filtered': None,
        'filters_applied': {},
        'debug_mode': False,
        'preprocessor': None,
        'label_encoder': None,
        'feature_names': None,
        'cleaning_report': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# =============================================================================
# DATA CLEANING FUNCTIONS
# =============================================================================

def parse_sf_date(date_str):
    """
    Parse San Francisco permit dates with multiple format support
    Handles: MM-DD-YY, MM/DD/YYYY, MM-DD-YYYY, etc.
    """
    if pd.isna(date_str) or date_str == '':
        return pd.NaT
    
    date_str = str(date_str).strip()
    
    # List of date formats to try (in order of likelihood)
    formats = [
        '%m-%d-%y',      # 05-06-15
        '%m/%d/%Y',      # 04/19/2016
        '%m-%d-%Y',      # 11-07-2016
        '%m/%d/%y',      # 06/30/17
        '%Y-%m-%d',      # 2016-05-27
        '%d-%m-%Y',      # 06-05-2015
        '%d/%m/%Y',      # 06/05/2015
    ]
    
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except (ValueError, TypeError):
            continue
    
    # If all formats fail, try pandas inference as last resort
    try:
        return pd.to_datetime(date_str, infer_datetime_format=True)
    except:
        return pd.NaT


def fix_permit_number(permit_num):
    """
    Fix permit numbers from scientific notation
    Handles: 2.01505E+11 -> 201505000000 (approximately)
    """
    if pd.isna(permit_num):
        return permit_num
    
    permit_str = str(permit_num).upper().strip()
    
    # Check if it's in scientific notation
    if 'E+' in permit_str or 'E-' in permit_str:
        try:
            # Convert scientific notation to integer
            num = float(permit_str)
            # Format as integer (no decimals)
            return f"{int(num)}"
        except (ValueError, OverflowError):
            return permit_str
    
    # If it starts with letters (like M803667), keep as is
    if permit_str and permit_str[0].isalpha():
        return permit_str
    
    return permit_str


@timing_decorator
def comprehensive_data_cleaning(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Comprehensive data cleaning for SF Building Permits
    Returns: (cleaned_dataframe, cleaning_report)
    """
    df_clean = df.copy()
    report = {
        'original_rows': len(df),
        'issues_fixed': [],
        'warnings': []
    }
    
    if verbose:
        with st.spinner("🔧 Starting comprehensive data cleaning..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # ===================================================================
            # 1. FIX PERMIT NUMBERS (Critical - do this first!)
            # ===================================================================
            status_text.text("📋 Fixing Permit Numbers...")
            progress_bar.progress(10)
            
            if 'Permit Number' in df_clean.columns:
                original_sci = df_clean['Permit Number'].astype(str).str.contains('E', case=False, na=False).sum()
                
                # Apply fix
                df_clean['Permit Number'] = df_clean['Permit Number'].apply(fix_permit_number)
                
                if original_sci > 0:
                    report['issues_fixed'].append(f"Fixed {original_sci} permit numbers from scientific notation")
            
            # ===================================================================
            # 2. FIX ALL DATE COLUMNS
            # ===================================================================
            status_text.text("📅 Fixing Date Columns...")
            progress_bar.progress(30)
            
            date_fixes = 0
            for col in Config.DATE_COLUMNS:
                if col in df_clean.columns:
                    original_valid = df_clean[col].notna().sum()
                    
                    # Apply custom date parser
                    df_clean[col] = df_clean[col].apply(parse_sf_date)
                    
                    new_valid = df_clean[col].notna().sum()
                    parsed = new_valid - (original_valid - df[col].isna().sum())
                    
                    if parsed > 0:
                        date_fixes += parsed
            
            if date_fixes > 0:
                report['issues_fixed'].append(f"Successfully parsed {date_fixes} dates across all date columns")
            
            # ===================================================================
            # 3. FIX NUMERIC COLUMNS
            # ===================================================================
            status_text.text("💰 Fixing Numeric Columns...")
            progress_bar.progress(50)
            
            numeric_columns = ['Estimated Cost', 'Revised Cost', 'Number of Existing Stories',
                             'Number of Proposed Stories', 'Existing Units', 'Proposed Units',
                             'Plansets']
            
            for col in numeric_columns:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    
                    # Handle negative values in cost columns
                    if 'Cost' in col:
                        negative_count = (df_clean[col] < 0).sum()
                        if negative_count > 0:
                            df_clean.loc[df_clean[col] < 0, col] = np.nan
                            report['issues_fixed'].append(f"Removed {negative_count} negative values from {col}")
            
            # ===================================================================
            # 4. STANDARDIZE STATUS VALUES
            # ===================================================================
            status_text.text("📊 Standardizing Status Values...")
            progress_bar.progress(70)
            
            if 'Current Status' in df_clean.columns:
                # Convert to lowercase and strip whitespace
                df_clean['Current Status'] = df_clean['Current Status'].str.lower().str.strip()
                
                # Check for invalid statuses
                invalid_status = ~df_clean['Current Status'].isin(Config.VALID_STATUSES)
                invalid_count = invalid_status.sum()
                
                if invalid_count > 0:
                    unique_invalid = df_clean[invalid_status]['Current Status'].unique()
                    report['warnings'].append(f"{invalid_count} permits with unexpected status values: {list(unique_invalid)[:5]}")
                else:
                    report['issues_fixed'].append("All status values are valid")
            
            # ===================================================================
            # 5. FIX SUPERVISOR DISTRICTS
            # ===================================================================
            status_text.text("🗺️ Validating Supervisor Districts...")
            progress_bar.progress(80)
            
            if 'Supervisor District' in df_clean.columns:
                df_clean['Supervisor District'] = pd.to_numeric(df_clean['Supervisor District'], errors='coerce')
                invalid_districts = ((df_clean['Supervisor District'] < 1) | 
                                   (df_clean['Supervisor District'] > 11)).sum()
                
                if invalid_districts > 0:
                    report['warnings'].append(f"{invalid_districts} permits with invalid district numbers")
            
            # ===================================================================
            # FINALIZE
            # ===================================================================
            status_text.text("✅ Data cleaning complete!")
            progress_bar.progress(100)
            
            report['final_rows'] = len(df_clean)
            report['rows_removed'] = report['original_rows'] - report['final_rows']
            
            # Clear progress indicators
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
    else:
        # Non-verbose mode - just do the cleaning
        if 'Permit Number' in df_clean.columns:
            df_clean['Permit Number'] = df_clean['Permit Number'].apply(fix_permit_number)
        
        for col in Config.DATE_COLUMNS:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(parse_sf_date)
        
        if 'Current Status' in df_clean.columns:
            df_clean['Current Status'] = df_clean['Current Status'].str.lower().str.strip()
        
        report['final_rows'] = len(df_clean)
        report['rows_removed'] = report['original_rows'] - report['final_rows']
    
    return df_clean, report


def display_cleaning_report(report: Dict):
    """Display the data cleaning report"""
    st.success(f"✅ Data Cleaning Complete!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Records", f"{report['original_rows']:,}")
    with col2:
        st.metric("Final Records", f"{report['final_rows']:,}")
    with col3:
        st.metric("Records Removed", f"{report['rows_removed']:,}")
    
    if report['issues_fixed']:
        with st.expander("✅ Issues Fixed", expanded=False):
            for fix in report['issues_fixed']:
                st.write(f"• {fix}")
    
    if report['warnings']:
        with st.expander("⚠️ Warnings", expanded=False):
            for warning in report['warnings']:
                st.write(f"• {warning}")

# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate dataframe structure and content for SF Building Permits
    Returns: (is_valid, list_of_issues)
    """
    issues = []
    
    # 1. Check required columns
    missing_cols = [col for col in Config.REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        return (False, issues)  # Critical failure
    
    # 2. Check for completely empty dataframe
    if df.empty:
        issues.append("DataFrame is empty")
        return (False, issues)
    
    # 3. Check for required fields that are mostly empty
    for col in Config.REQUIRED_COLUMNS:
        if col in df.columns:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            if missing_pct > 50:
                issues.append(f"WARNING: '{col}' is {missing_pct:.1f}% empty")
    
    # Return validation result
    is_valid = len(issues) == 0
    return (is_valid, issues)


def detect_outliers(series: pd.Series, method: str = 'iqr', threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers in a numeric series
    
    Args:
        series: Numeric pandas Series
        method: 'iqr' or 'zscore'
        threshold: Multiplier for IQR or z-score threshold
    
    Returns:
        Boolean series indicating outliers
    """
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    return pd.Series([False] * len(series), index=series.index)

# =============================================================================
# DATA PREPROCESSING
# =============================================================================

@timing_decorator
def drop_high_missing_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop columns with missing rate above threshold"""
    missing_ratio = df.isnull().sum() / len(df)
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    
    if cols_to_drop and st.session_state.get('debug_mode', False):
        st.info(f"Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing values")
    
    return df.drop(columns=cols_to_drop)

@timing_decorator
def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that won't help prediction"""
    cols_to_drop = [col for col in Config.IRRELEVANT_COLS if col in df.columns]
    
    if cols_to_drop and st.session_state.get('debug_mode', False):
        st.info(f"Dropped {len(cols_to_drop)} irrelevant columns")
    
    return df.drop(columns=cols_to_drop, errors='ignore')

@timing_decorator
def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer temporal features from date columns"""
    
    # Processing time features
    if {'Filed Date', 'Issued Date'}.issubset(df.columns):
        df['Processing_Days'] = (df['Issued Date'] - df['Filed Date']).dt.days
        df['Processing_Days'] = df['Processing_Days'].clip(lower=0, upper=365*5)  # Cap at 5 years
        
        # Flag suspicious processing times
        df['Processing_Suspicious'] = (
            (df['Processing_Days'] < 0) | 
            (df['Processing_Days'] > 365*2)
        ).astype(int)
    
    # Extract temporal features from Filed Date
    if 'Filed Date' in df.columns:
        df['Filed_Year'] = df['Filed Date'].dt.year
        df['Filed_Month'] = df['Filed Date'].dt.month
        df['Filed_Quarter'] = df['Filed Date'].dt.quarter
        df['Filed_DayOfWeek'] = df['Filed Date'].dt.dayofweek
        df['Filed_IsWeekend'] = (df['Filed_DayOfWeek'] >= 5).astype(int)
        
        # Season (construction seasonality)
        df['Filed_Season'] = df['Filed_Month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
    
    # Extract features from Permit Expiration Date
    if 'Permit Expiration Date' in df.columns:
        df['Expiration_Year'] = df['Permit Expiration Date'].dt.year
        df['Expiration_Month'] = df['Permit Expiration Date'].dt.month
        
        # Permit duration (Filed to Expiration)
        if 'Filed Date' in df.columns:
            df['Permit_Duration_Days'] = (df['Permit Expiration Date'] - df['Filed Date']).dt.days
            df['Permit_Duration_Days'] = df['Permit_Duration_Days'].clip(lower=0)
    
    return df

@timing_decorator
def engineer_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features related to changes in units, stories, etc."""
    
    # Unit change features
    if {'Existing Units', 'Proposed Units'}.issubset(df.columns):
        df['Unit_Change'] = df['Proposed Units'] - df['Existing Units']
        df['Unit_Change_Pct'] = ((df['Proposed Units'] - df['Existing Units']) /
                                (df['Existing Units'] + 1)) * 100
        df['Is_Adding_Units'] = (df['Unit_Change'] > 0).astype(int)
    
    # Story change features
    if {'Number of Existing Stories', 'Number of Proposed Stories'}.issubset(df.columns):
        df['Story_Change'] = df['Number of Proposed Stories'] - df['Number of Existing Stories']
        df['Is_Adding_Stories'] = (df['Story_Change'] > 0).astype(int)
    
    # Construction type change
    if {'Existing Construction Type', 'Proposed Construction Type'}.issubset(df.columns):
        df['Construction_Type_Change'] = (
            df['Existing Construction Type'] != df['Proposed Construction Type']
        ).astype(int)
    
    # Use change
    if {'Existing Use', 'Proposed Use'}.issubset(df.columns):
        df['Use_Change'] = (df['Existing Use'] != df['Proposed Use']).astype(int)
    
    return df

@timing_decorator
def engineer_cost_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer cost-related features"""
    
    if 'Estimated Cost' in df.columns:
        # Log transform for better distribution
        df['Log_Estimated_Cost'] = np.log1p(df['Estimated Cost'])
        
        # Cost per unit
        if 'Proposed Units' in df.columns:
            df['Cost_Per_Unit'] = df['Estimated Cost'] / (df['Proposed Units'] + 1)
            df['Log_Cost_Per_Unit'] = np.log1p(df['Cost_Per_Unit'])
        
        # High cost flag
        df['High_Cost_Project'] = (
            df['Estimated Cost'] > df['Estimated Cost'].quantile(0.75)
        ).astype(int)
    
    return df

@timing_decorator
def engineer_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features for better predictions"""
    
    # Project complexity score
    complexity_components = []
    
    if 'Story_Change' in df.columns:
        complexity_components.append(df['Story_Change'].abs())
    
    if 'Unit_Change' in df.columns:
        complexity_components.append(df['Unit_Change'].abs())
    
    if 'Construction_Type_Change' in df.columns:
        complexity_components.append(df['Construction_Type_Change'] * 5)
    
    if complexity_components:
        df['Project_Complexity'] = sum(complexity_components)
    
    return df

@timing_decorator
def handle_outliers(df: pd.DataFrame, method: str = 'cap', threshold: float = 3.0) -> pd.DataFrame:
    """Handle outliers in numerical features"""
    
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude certain columns
    exclude_cols = ['Filed_Year', 'Filed_Month', 'Filed_Quarter', 'Filed_DayOfWeek']
    numerical_features = [col for col in numerical_features if col not in exclude_cols]
    
    outlier_counts = {}
    
    for col in numerical_features:
        outliers = detect_outliers(df[col].dropna(), method='iqr', threshold=threshold)
        outliers_count = outliers.sum()
        
        if outliers_count > 0:
            outlier_counts[col] = outliers_count
            
            if method == 'cap':
                # Cap outliers
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    if outlier_counts and st.session_state.get('debug_mode', False):
        st.sidebar.write("Outliers handled:", outlier_counts)
    
    return df

@timing_decorator
def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary target variable for classification"""
    
    if 'Current Status' in df.columns:
        df['Status_Cleaned'] = df['Current Status'].apply(
            lambda x: 'Inactive' if any(status in str(x).lower().strip() 
                                       for status in Config.INACTIVE_STATUSES)
            else 'Active'
        )
    
    return df

@timing_decorator
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete preprocessing pipeline
    """
    with st.spinner("🔄 Preprocessing data..."):
        # 1. Drop high-missing columns
        df = drop_high_missing_columns(df, threshold=0.5)
        
        # 2. Drop irrelevant columns
        df = drop_irrelevant_columns(df)
        
        # 3. Engineer features
        df = engineer_temporal_features(df)
        df = engineer_change_features(df)
        df = engineer_cost_features(df)
        df = engineer_interaction_features(df)
        
        # 4. Create target variable
        df = create_target_variable(df)
        
        # 5. Handle outliers
        df = handle_outliers(df)
        
        # 6. Drop original date columns (we've extracted features)
        date_columns_to_drop = [col for col in Config.DATE_COLUMNS if col in df.columns]
        df = df.drop(columns=date_columns_to_drop, errors='ignore')
    
    st.success("✅ Preprocessing complete!")
    
    return df

# =============================================================================
# DATA LOADING
# =============================================================================

@timing_decorator
@st.cache_data(ttl=Config.CACHE_TTL, show_spinner=False)
def load_data(file_path=None, uploaded_file=None) -> Optional[pd.DataFrame]:
    """Load and cache the building permits dataset"""
    
    try:
        if uploaded_file is not None:
            # Read uploaded file
            chunks = []
            
            with st.spinner("Loading data..."):
                for chunk in pd.read_csv(
                    uploaded_file,
                    dtype=str,  # Load as strings to preserve permit numbers
                    low_memory=False,
                    chunksize=Config.CHUNK_SIZE
                ):
                    chunks.append(chunk)
            
            df = pd.concat(chunks, ignore_index=True)
            
        elif file_path:
            # Read from file path
            chunks = []
            
            with st.spinner("Loading data..."):
                for chunk in pd.read_csv(
                    file_path,
                    dtype=str,  # Load as strings to preserve permit numbers
                    low_memory=False,
                    chunksize=Config.CHUNK_SIZE
                ):
                    chunks.append(chunk)
            
            df = pd.concat(chunks, ignore_index=True)
        else:
            st.warning("No data file found. Please upload 'CleanedData.csv'")
            return None
        
        # Apply comprehensive data cleaning
        df_clean, cleaning_report = comprehensive_data_cleaning(df, verbose=True)
        
        # Store cleaning report in session state
        st.session_state.cleaning_report = cleaning_report
        
        # Display cleaning report
        display_cleaning_report(cleaning_report)
        
        # Validate cleaned data
        is_valid, issues = validate_dataframe(df_clean)
        
        if not is_valid:
            st.warning("⚠️ Data validation issues detected after cleaning:")
            for issue in issues:
                st.write(f"- {issue}")
        
        # Apply preprocessing pipeline
        df_clean = preprocess_data(df_clean)
        
        return df_clean
    
    except FileNotFoundError:
        st.error(f"❌ File not found: {file_path}")
        return None
    
    except pd.errors.EmptyDataError:
        st.error("❌ The file is empty")
        return None
    
    except pd.errors.ParserError as e:
        st.error(f"❌ Error parsing CSV file: {str(e)}")
        return None
    
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        if st.session_state.get('debug_mode', False):
            st.exception(e)
        return None

# =============================================================================
# STATISTICS AND CALCULATIONS
# =============================================================================

@timing_decorator
@st.cache_data(show_spinner=False)
def calculate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate key statistics from the dataset"""
    
    # Create binary status classification
    status_cleaned = None
    if 'Status_Cleaned' in df.columns:
        status_cleaned = df['Status_Cleaned']
    
    # Calculate statistics
    stats = {
        'total_permits': len(df),
        'total_columns': len(df.columns),
        'memory_usage': df.memory_usage(deep=True).sum() / (1024**2),
        'date_range': (
            df['Filed_Year'].min() if 'Filed_Year' in df.columns else None,
            df['Filed_Year'].max() if 'Filed_Year' in df.columns else None
        ),
        'unique_neighborhoods': (
            df['Neighborhoods - Analysis Boundaries'].nunique() 
            if 'Neighborhoods - Analysis Boundaries' in df.columns else 0
        ),
        'active_permits': (status_cleaned == 'Active').sum() if status_cleaned is not None else 0,
        'inactive_permits': (status_cleaned == 'Inactive').sum() if status_cleaned is not None else 0,
        'class_balance_ratio': None
    }
    
    # Calculate class balance ratio
    if status_cleaned is not None:
        status_counts = status_cleaned.value_counts()
        if len(status_counts) == 2:
            stats['class_balance_ratio'] = status_counts.min() / status_counts.max()
    
    return stats

# =============================================================================
# FILTERING
# =============================================================================

@st.cache_data(show_spinner=False)
def filter_dataframe(
    df: pd.DataFrame,
    year_range: Optional[Tuple],
    status: str,
    permit_type: str
) -> pd.DataFrame:
    """Filter dataframe based on user selections"""
    
    df_filtered = df.copy()
    
    # Year filter
    if year_range and len(year_range) == 2 and 'Filed_Year' in df.columns:
        df_filtered = df_filtered[
            (df_filtered['Filed_Year'] >= year_range[0]) &
            (df_filtered['Filed_Year'] <= year_range[1])
        ]
    
    # Status filter
    if status != 'All' and 'Current Status' in df.columns:
        df_filtered = df_filtered[df_filtered['Current Status'] == status]
    
    # Permit type filter
    if permit_type != 'All' and 'Permit Type' in df.columns:
        df_filtered = df_filtered[df_filtered['Permit Type'] == permit_type]
    
    return df_filtered

def create_advanced_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Create advanced filtering sidebar"""
    
    with st.sidebar.expander("🔍 Advanced Filters", expanded=False):
        
        # Cost range
        if 'Estimated Cost' in df.columns:
            cost_data = df['Estimated Cost'].dropna()
            if len(cost_data) > 0:
                cost_min = float(cost_data.min())
                cost_max = float(cost_data.max())
                
                cost_range = st.slider(
                    "Cost Range ($)",
                    min_value=cost_min,
                    max_value=cost_max,
                    value=(cost_min, cost_max),
                    format="$%.0f"
                )
                df = df[
                    (df['Estimated Cost'].isna()) |
                    ((df['Estimated Cost'] >= cost_range[0]) & 
                     (df['Estimated Cost'] <= cost_range[1]))
                ]
        
        # Number of stories
        if 'Number of Proposed Stories' in df.columns:
            stories_data = df['Number of Proposed Stories'].dropna()
            if len(stories_data) > 0:
                stories_min = int(stories_data.min())
                stories_max = int(stories_data.max())
                
                stories_range = st.slider(
                    "Number of Stories",
                    min_value=stories_min,
                    max_value=stories_max,
                    value=(stories_min, stories_max)
                )
                df = df[
                    (df['Number of Proposed Stories'].isna()) |
                    ((df['Number of Proposed Stories'] >= stories_range[0]) &
                     (df['Number of Proposed Stories'] <= stories_range[1]))
                ]
        
        # Neighborhood filter
        if 'Neighborhoods - Analysis Boundaries' in df.columns:
            neighborhoods = ['All'] + sorted(
                df['Neighborhoods - Analysis Boundaries'].dropna().unique().tolist()
            )
            selected_neighborhood = st.selectbox("Neighborhood", neighborhoods)
            
            if selected_neighborhood != 'All':
                df = df[df['Neighborhoods - Analysis Boundaries'] == selected_neighborhood]
    
    return df

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_plotly_figure(
    fig: go.Figure,
    title: str,
    height: int = 500
) -> go.Figure:
    """Apply consistent styling to plotly figures"""
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center', font=dict(size=18)),
        height=height,
        template='plotly_white',
        font=dict(size=12),
        hovermode='closest',
        showlegend=True,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

@timing_decorator
def plot_missing_values(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create a visualization of missing values"""
    
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    
    if len(missing_pct) == 0:
        return None
    
    fig = px.bar(
        x=missing_pct.values,
        y=missing_pct.index,
        orientation='h',
        labels={'x': 'Percentage Missing (%)', 'y': 'Column'},
        color=missing_pct.values,
        color_continuous_scale='Reds'
    )
    
    fig = create_plotly_figure(fig, 'Missing Values by Column (%)', 
                               height=max(400, len(missing_pct) * 20))
    fig.update_layout(showlegend=False)
    
    return fig

@timing_decorator
def plot_permit_status_distribution(df: pd.DataFrame) -> Optional[go.Figure]:
    """Plot distribution of permit statuses"""
    
    if 'Status_Cleaned' not in df.columns:
        return None
    
    status_counts = df['Status_Cleaned'].value_counts()
    
    fig = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        hole=0.4,
        color_discrete_sequence=['#2ecc71', '#e74c3c']
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig = create_plotly_figure(fig, 'Distribution of Permit Statuses')
    
    return fig

@timing_decorator
def plot_permits_over_time(df: pd.DataFrame) -> Optional[go.Figure]:
    """Plot permits filed over time with trend line"""
    
    if 'Filed_Year' not in df.columns or 'Filed_Month' not in df.columns:
        return None
    
    df_time = df[(df['Filed_Year'].notna()) & (df['Filed_Month'].notna())].copy()
    df_time['YearMonth'] = df_time['Filed_Year'].astype(str) + '-' + df_time['Filed_Month'].astype(str).str.zfill(2)
    monthly_counts = df_time.groupby('YearMonth').size().reset_index(name='Count')
    monthly_counts = monthly_counts.sort_values('YearMonth')
    
    # Create figure
    fig = go.Figure()
    
    # Actual data
    fig.add_trace(go.Scatter(
        x=monthly_counts['YearMonth'],
        y=monthly_counts['Count'],
        mode='lines+markers',
        name='Permits Filed',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4)
    ))
    
    # Moving average
    window = 6
    if len(monthly_counts) >= window:
        monthly_counts['MA'] = monthly_counts['Count'].rolling(window=window, center=True).mean()
        
        fig.add_trace(go.Scatter(
            x=monthly_counts['YearMonth'],
            y=monthly_counts['MA'],
            mode='lines',
            name=f'{window}-Month Moving Average',
            line=dict(color='red', width=2, dash='dash')
        ))
    
    fig.update_xaxes(tickangle=45, nticks=20, title='Month')
    fig.update_yaxes(title='Number of Permits')
    fig.update_layout(hovermode='x unified')
    
    fig = create_plotly_figure(fig, 'Permits Filed Over Time with Trend', height=500)
    
    return fig

@timing_decorator
def plot_permits_by_type(df: pd.DataFrame) -> Optional[go.Figure]:
    """Plot distribution of permit types"""
    
    if 'Permit Type Definition' not in df.columns:
        return None
    
    type_counts = df['Permit Type Definition'].value_counts().head(15)
    
    fig = px.bar(
        x=type_counts.values,
        y=type_counts.index,
        orientation='h',
        labels={'x': 'Number of Permits', 'y': 'Permit Type Definition'},
        color=type_counts.values,
        color_continuous_scale='Blues'
    )
    
    fig = create_plotly_figure(fig, 'Top 15 Permit Types', height=500)
    fig.update_layout(showlegend=False)
    
    return fig

@timing_decorator
def plot_cost_distribution(df: pd.DataFrame) -> Optional[go.Figure]:
    """Plot distribution of estimated costs"""
    
    if 'Estimated Cost' not in df.columns:
        return None
    
    df_cost = df[df['Estimated Cost'].notna() & (df['Estimated Cost'] > 0)].copy()
    
    if len(df_cost) == 0:
        return None
    
    df_cost['Log Cost'] = np.log10(df_cost['Estimated Cost'] + 1)
    
    fig = px.histogram(
        df_cost,
        x='Log Cost',
        nbins=50,
        labels={'Log Cost': 'Log10(Estimated Cost)'},
        marginal='box'
    )
    
    fig = create_plotly_figure(fig, 'Distribution of Estimated Costs (Log Scale)', height=500)
    
    return fig

@timing_decorator
def plot_neighborhood_analysis(df: pd.DataFrame) -> Optional[go.Figure]:
    """Analyze permits by neighborhood"""
    
    if 'Neighborhoods - Analysis Boundaries' not in df.columns:
        return None
    
    neighborhood_counts = df['Neighborhoods - Analysis Boundaries'].value_counts().head(20)
    
    fig = px.bar(
        x=neighborhood_counts.values,
        y=neighborhood_counts.index,
        orientation='h',
        labels={'x': 'Number of Permits', 'y': 'Neighborhood'},
        color=neighborhood_counts.values,
        color_continuous_scale='Viridis'
    )
    
    fig = create_plotly_figure(fig, 'Top 20 Neighborhoods by Number of Permits', height=600)
    fig.update_layout(showlegend=False)
    
    return fig

@timing_decorator
def plot_processing_time(df: pd.DataFrame) -> Optional[go.Figure]:
    """Analyze permit processing time"""
    
    if 'Processing_Days' not in df.columns:
        return None
    
    df_time = df[df['Processing_Days'].notna()].copy()
    df_time = df_time[df_time['Processing_Days'] >= 0]
    df_time = df_time[df_time['Processing_Days'] <= df_time['Processing_Days'].quantile(0.95)]
    
    if len(df_time) == 0:
        return None
    
    fig = px.histogram(
        df_time,
        x='Processing_Days',
        nbins=50,
        labels={'Processing_Days': 'Days'},
        marginal='box'
    )
    
    fig = create_plotly_figure(
        fig, 
        'Distribution of Permit Processing Time (Days from Filed to Issued)',
        height=500
    )
    
    return fig

@timing_decorator
def plot_construction_types(df: pd.DataFrame) -> Optional[go.Figure]:
    """Analyze proposed construction types"""
    
    if 'Proposed Construction Type Description' not in df.columns:
        return None
    
    const_types = df['Proposed Construction Type Description'].value_counts()
    
    if len(const_types) == 0:
        return None
    
    fig = px.pie(
        values=const_types.values,
        names=const_types.index,
        hole=0.3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig = create_plotly_figure(fig, 'Distribution of Proposed Construction Types')
    
    return fig

@timing_decorator
def plot_stories_analysis(df: pd.DataFrame) -> Optional[go.Figure]:
    """Analyze building stories"""
    
    if not {'Number of Proposed Stories', 'Number of Existing Stories'}.issubset(df.columns):
        return None
    
    df_stories = df[
        (df['Number of Proposed Stories'].notna()) & 
        (df['Number of Existing Stories'].notna())
    ].copy()
    
    df_stories = df_stories[df_stories['Number of Proposed Stories'] <= 20]
    
    if len(df_stories) == 0:
        return None
    
    fig = px.scatter(
        df_stories,
        x='Number of Existing Stories',
        y='Number of Proposed Stories',
        labels={
            'Number of Existing Stories': 'Existing Stories',
            'Number of Proposed Stories': 'Proposed Stories'
        },
        opacity=0.5
    )
    
    # Add diagonal line
    max_val = max(
        df_stories['Number of Existing Stories'].max(),
        df_stories['Number of Proposed Stories'].max()
    )
    
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='No Change',
        line=dict(color='red', dash='dash')
    ))
    
    fig = create_plotly_figure(fig, 'Existing vs Proposed Number of Stories', height=500)
    
    return fig

@timing_decorator
def plot_supervisor_district(df: pd.DataFrame) -> Optional[go.Figure]:
    """Analyze permits by supervisor district"""
    
    if 'Supervisor District' not in df.columns:
        return None
    
    district_counts = df['Supervisor District'].value_counts().sort_index()
    
    fig = px.bar(
        x=district_counts.index.astype(str),
        y=district_counts.values,
        labels={'x': 'Supervisor District', 'y': 'Number of Permits'},
        color=district_counts.values,
        color_continuous_scale='Teal'
    )
    
    fig = create_plotly_figure(fig, 'Number of Permits by Supervisor District', height=500)
    fig.update_layout(showlegend=False)
    
    return fig

# =============================================================================
# MACHINE LEARNING FUNCTIONS
# =============================================================================

@timing_decorator
def prepare_ml_data(df: pd.DataFrame, test_size: float = 0.2, use_smote: bool = True):
    """Prepare data for machine learning"""
    
    if 'Status_Cleaned' not in df.columns:
        st.error("Target variable 'Status_Cleaned' not found in dataset")
        return None
    
    # Remove rows with missing target
    df_ml = df[df['Status_Cleaned'].notna()].copy()
    
    # Separate features and target
    X = df_ml.drop(columns=['Status_Cleaned', 'Current Status'], errors='ignore')
    y = df_ml['Status_Cleaned']
    
    # Identify column types
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Split data FIRST to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=Config.RANDOM_STATE, stratify=y
    )
    
    # Handle missing values
    knn_imputer = KNNImputer(n_neighbors=5)
    if numerical_cols:
        X_train[numerical_cols] = knn_imputer.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = knn_imputer.transform(X_test[numerical_cols])
    
    cat_imputer = SimpleImputer(strategy='most_frequent')
    if categorical_cols:
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])
    
    # Clip negative values for positive features
    positive_features = [
        'Estimated Cost', 'Existing Units', 'Proposed Units',
        'Number of Existing Stories', 'Number of Proposed Stories',
        'Plansets'
    ]
    positive_features = [f for f in positive_features if f in numerical_cols]
    
    for col in positive_features:
        if (X_train[col] < 0).any():
            X_train[col] = X_train[col].clip(lower=0)
            X_test[col] = X_test[col].clip(lower=0)
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Handle class imbalance with SMOTE
    if use_smote:
        class_counts = pd.Series(y_train_encoded).value_counts()
        imbalance_ratio = class_counts.min() / class_counts.max()
        
        if imbalance_ratio < 0.5:  # If minority class is less than 50% of majority
            smote = SMOTE(random_state=Config.RANDOM_STATE)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train_encoded)
        else:
            X_train_balanced = X_train_processed
            y_train_balanced = y_train_encoded
    else:
        X_train_balanced = X_train_processed
        y_train_balanced = y_train_encoded
    
    # Get feature names
    feature_names = numerical_cols.copy()
    if categorical_cols:
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
        feature_names.extend(cat_features)
    
    return {
        'X_train': X_train_balanced,
        'X_test': X_test_processed,
        'y_train': y_train_balanced,
        'y_test': y_test_encoded,
        'preprocessor': preprocessor,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'class_distribution': pd.Series(y_train_balanced).value_counts().to_dict()
    }

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str) -> go.Figure:
    """Plot confusion matrix as heatmap"""
    
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels,
        y=labels,
        color_continuous_scale='Blues'
    )
    
    fig = create_plotly_figure(fig, title, height=400)
    
    return fig

# =============================================================================
# DATA QUALITY DASHBOARD
# =============================================================================

def create_data_quality_report(df: pd.DataFrame):
    """Create comprehensive data quality dashboard"""
    
    st.header("📊 Data Quality Report")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        st.metric("Completeness", f"{completeness:.1f}%")
    
    with col2:
        duplicates = df.duplicated().sum()
        st.metric("Duplicate Rows", duplicates)
    
    with col3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = sum([detect_outliers(df[col].dropna()).sum() for col in numeric_cols])
        st.metric("Outliers Detected", outliers)
    
    with col4:
        valid, issues = validate_dataframe(df)
        st.metric("Data Issues", len(issues))
    
    # Show cleaning report if available
    if st.session_state.cleaning_report:
        st.markdown("---")
        st.subheader("🔧 Data Cleaning Summary")
        display_cleaning_report(st.session_state.cleaning_report)
    
    # Detailed issues
    if issues:
        st.warning("⚠️ Data Quality Issues Detected")
        for issue in issues:
            st.write(f"- {issue}")
    else:
        st.success("✅ No data quality issues detected")
    
    # Missing values heatmap
    st.subheader("Missing Values Analysis")
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    
    if len(missing_pct) > 0:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig = plot_missing_values(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### High Missing Columns")
            high_missing = missing_pct[missing_pct > 20]
            for col, pct in high_missing.items():
                st.write(f"**{col}**: {pct:.1f}%")
    else:
        st.success("No missing values detected!")

# =============================================================================
# EXPORT FUNCTIONALITY
# =============================================================================

def create_export_section(df: pd.DataFrame):
    """Create comprehensive export section"""
    
    st.markdown("### 📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV Export
        csv = df.to_csv(index=False)
        st.download_button(
            label="📄 Download as CSV",
            data=csv,
            file_name=f"permits_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel Export
        buffer = io.BytesIO()
        try:
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Limit to 100k rows for Excel
                df.head(100000).to_excel(writer, sheet_name='Data', index=False)
                df.describe().to_excel(writer, sheet_name='Statistics')
            
            st.download_button(
                label="📊 Download as Excel",
                data=buffer.getvalue(),
                file_name=f"permits_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except ImportError:
            st.info("Install openpyxl to enable Excel export")
    
    with col3:
        # JSON Export (limited to 1000 rows)
        json_str = df.head(1000).to_json(orient='records', date_format='iso')
        st.download_button(
            label="📋 Download as JSON",
            data=json_str,
            file_name=f"permits_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point"""
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown(
        '<p class="main-header">🏗️ San Francisco Building Permits Analysis</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Interactive Dashboard for Building Permit Data Exploration & ML Predictions</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    st.sidebar.title("🎛️ Navigation")
    st.sidebar.markdown("---")
    
    # Debug mode toggle
    st.sidebar.checkbox("Debug Mode", key='debug_mode')
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload Building Permits CSV",
        type=['csv'],
        help="Upload the building permits CSV file"
    )
    
    # Load data
    df = None
    
    if uploaded_file is not None:
        df = load_data(uploaded_file=uploaded_file)
    else:
        # Try to load from default location
        try:
            df = load_data(file_path="CleanedData.csv")
        except:
            pass
    
    if df is None:
        st.info("👆 Please upload the Building Permits CSV file using the sidebar to begin analysis.")
        
        # Show sample data structure
        st.markdown("### Expected CSV Structure")
        st.markdown("""
        The CSV file should contain columns like:
        - Permit Number
        - Permit Type
        - Current Status
        - Filed Date
        - Issued Date
        - Estimated Cost
        - Neighborhoods
        - And more...
        """)
        st.stop()

    # Store original dataframe - ADD THIS CHECK
    if df is not None and st.session_state.df_original is None:
        st.session_state.df_original = df.copy()

    # Also add a safety check here
    if df is None:
        st.stop()

    # Calculate statistics
    stats = calculate_statistics(df)
    
    # Sidebar filters
    st.sidebar.markdown("### 🔍 Filters")
    
    # Year range filter
    year_range = None
    if 'Filed_Year' in df.columns and stats['date_range'][0] is not None:
        year_min = int(stats['date_range'][0])
        year_max = int(stats['date_range'][1])
        
        year_range = st.sidebar.slider(
            "Select Year Range",
            min_value=year_min,
            max_value=year_max,
            value=(year_min, year_max)
        )
    
    # Status filter
    selected_status = 'All'
    if 'Current Status' in df.columns:
        statuses = ['All'] + sorted(df['Current Status'].dropna().unique().tolist())
        selected_status = st.sidebar.selectbox("Permit Status", statuses)
    
    # Permit type filter
    selected_type = 'All'
    if 'Permit Type' in df.columns:
        permit_types = ['All'] + sorted(df['Permit Type'].dropna().unique().tolist())
        selected_type = st.sidebar.selectbox("Permit Type", permit_types)
    
    # Apply filters
    if year_range or selected_status != 'All' or selected_type != 'All':
        df = filter_dataframe(df, year_range, selected_status, selected_type)
    
    # Advanced filters
    df = create_advanced_filters(df)
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"📊 Filtered records: **{len(df):,}** / {stats['total_permits']:,}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Temporal Analysis",
        "🏘️ Geographical Analysis",
        "💰 Cost & Construction",
        "📋 Data Explorer",
    ])

    # Tab 1: Overview
    with tab1:
        st.header("Dataset Overview")
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Permits", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", stats['total_columns'])
        with col3:
            st.metric("Memory Usage", f"{stats['memory_usage']:.2f} MB")
        with col4:
            st.metric("Neighborhoods", stats['unique_neighborhoods'])
        with col5:
            active_pct = (stats['active_permits'] / stats['total_permits'] * 100) if stats['total_permits'] > 0 else 0
            st.metric("Active Permits", f"{active_pct:.1f}%")
        
        st.markdown("---")
        
        # Two columns for visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Permit Status Distribution")
            fig = plot_permit_status_distribution(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No status data available")
        
        with col2:
            st.subheader("Top Permit Types")
            fig = plot_permits_by_type(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No permit type data available")
        create_data_quality_report(df)

    # Tab 2: Temporal Analysis
    with tab2:
        st.header("Temporal Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Permits Filed Over Time")
            fig = plot_permits_over_time(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No temporal data available")
        
        with col2:
            st.subheader("Key Statistics")
            if 'Filed_Year' in df.columns:
                st.metric("Earliest Year", int(df['Filed_Year'].min()))
                st.metric("Latest Year", int(df['Filed_Year'].max()))
                
                # Average permits per year
                years = df['Filed_Year'].max() - df['Filed_Year'].min() + 1
                avg_per_year = len(df) / years if years > 0 else 0
                st.metric("Avg Permits/Year", f"{avg_per_year:,.0f}")
        
        st.markdown("---")
        
        st.subheader("Permit Processing Time Analysis")
        fig = plot_processing_time(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # Processing time statistics
            if 'Processing_Days' in df.columns:
                df_time = df[df['Processing_Days'].notna()].copy()
                df_time = df_time[df_time['Processing_Days'] >= 0]
                
                if len(df_time) > 0:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Median Processing", f"{df_time['Processing_Days'].median():.0f} days")
                    with col2:
                        st.metric("Mean Processing", f"{df_time['Processing_Days'].mean():.0f} days")
                    with col3:
                        st.metric("Min Processing", f"{df_time['Processing_Days'].min():.0f} days")
                    with col4:
                        st.metric("95th Percentile", f"{df_time['Processing_Days'].quantile(0.95):.0f} days")
        else:
            st.warning("No processing time data available")
    
    # Tab 3: Geographical Analysis
    with tab3:
        st.header("Geographical Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Neighborhoods")
            fig = plot_neighborhood_analysis(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No neighborhood data available")
        
        with col2:
            st.subheader("Supervisor Districts")
            fig = plot_supervisor_district(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No district data available")
        
        # Zipcode analysis
        if 'Zipcode' in df.columns:
            st.markdown("---")
            st.subheader("Zipcode Distribution")
            
            zipcode_counts = df['Zipcode'].value_counts().head(15)
            fig = px.bar(
                x=zipcode_counts.index.astype(str),
                y=zipcode_counts.values,
                labels={'x': 'Zipcode', 'y': 'Number of Permits'},
                color=zipcode_counts.values,
                color_continuous_scale='Greens'
            )
            fig = create_plotly_figure(fig, 'Top 15 Zipcodes by Number of Permits', height=500)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Cost & Construction
    with tab4:
        st.header("Cost & Construction Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Estimated Cost Distribution")
            fig = plot_cost_distribution(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No cost data available")
            
            # Cost statistics
            if 'Estimated Cost' in df.columns:
                cost_data = df[df['Estimated Cost'].notna() & (df['Estimated Cost'] > 0)]['Estimated Cost']
                
                if len(cost_data) > 0:
                    st.markdown("### Cost Statistics")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Median Cost", f"${cost_data.median():,.0f}")
                    with col_b:
                        st.metric("Mean Cost", f"${cost_data.mean():,.0f}")
                    with col_c:
                        st.metric("Total Value", f"${cost_data.sum()/1e9:.2f}B")
        
        with col2:
            st.subheader("Construction Types")
            fig = plot_construction_types(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No construction type data available")
        
        st.markdown("---")
        
        st.subheader("Building Stories Analysis")
        fig = plot_stories_analysis(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # Stories statistics
            if 'Number of Proposed Stories' in df.columns:
                stories_data = df[df['Number of Proposed Stories'].notna()]['Number of Proposed Stories']
                
                if len(stories_data) > 0:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Avg Stories", f"{stories_data.mean():.1f}")
                    with col2:
                        st.metric("Median Stories", f"{stories_data.median():.0f}")
                    with col3:
                        st.metric("Max Stories", f"{stories_data.max():.0f}")
                    with col4:
                        single_story = (stories_data == 1).sum() / len(stories_data) * 100
                        st.metric("Single Story", f"{single_story:.1f}%")
        else:
            st.warning("No stories data available")
    
    # Tab 5: Data Explorer
    with tab5:
        st.header("Data Explorer")
        
        st.markdown("### Dataset Preview")
        
        # Show number of records
        st.info(f"Showing up to 100 records from {len(df):,} total (after filtering)")
        
        # Column selector
        all_columns = df.columns.tolist()
        default_columns = [
            col for col in ['Permit Type', 'Status_Cleaned',
                           'Neighborhoods - Analysis Boundaries',
                           'Estimated Cost', 'Filed_Year']
            if col in all_columns
        ]
        
        selected_columns = st.multiselect(
            "Select columns to display",
            options=all_columns,
            default=default_columns if default_columns else all_columns[:5]
        )
        
        if selected_columns:
            st.dataframe(df[selected_columns].head(100), use_container_width=True)
        else:
            st.dataframe(df.head(100), use_container_width=True)
        
        # Export section
        st.markdown("---")
        create_export_section(df)
        
        # Basic statistics
        st.markdown("---")
        st.markdown("### Statistical Summary")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        else:
            st.info("No numeric columns to display statistics")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p><strong>Building Permits Analysis Dashboard v3.1</strong> | San Francisco Open Data</p>
            <p>Built with Streamlit 🎈 | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Showing {len(df):,} of {stats['total_permits']:,} total permits</p>
        </div>
    """, unsafe_allow_html=True)

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()

