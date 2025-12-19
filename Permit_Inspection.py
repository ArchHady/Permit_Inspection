"""
Building Permits Analysis Dashboard
Interactive Streamlit application for San Francisco Building Permits data analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Page configuration
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
    </style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data(file_path=None):
    """Load and cache the building permits dataset"""
    try:
        if file_path:
            df = pd.read_csv(
                file_path,
                dtype={
                    'Voluntary Soft-Story Retrofit': str,
                    'TIDF Compliance': str
                },
                low_memory=False
            )
        else:
            # For demo purposes, create sample data if no file is provided
            st.warning("No data file found. Please upload 'CleanedData.csv' to view actual data.")
            return None

        # Convert date columns
        date_columns = [
            'Permit Creation Date', 'Current Status Date', 'Filed Date',
            'Issued Date', 'Completed Date', 'First Construction Document Date',
            'Permit Expiration Date'
        ]

        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def calculate_statistics(df):
    """Calculate key statistics from the dataset"""
    stats = {
        'total_permits': len(df),
        'total_columns': len(df.columns),
        'memory_usage': df.memory_usage(deep=True).sum() / (1024**2),
        'date_range': (df['Filed Date'].min(), df['Filed Date'].max()) if 'Filed Date' in df.columns else (None, None),
        'unique_neighborhoods': df['Neighborhoods - Analysis Boundaries'].nunique() if 'Neighborhoods - Analysis Boundaries' in df.columns else 0,
        'active_permits': len(df[df['Current Status'] == 'issued']) if 'Current Status' in df.columns else 0,
        'completed_permits': len(df[df['Current Status'] == 'complete']) if 'Current Status' in df.columns else 0,
    }
    return stats

def plot_missing_values(df):
    """Create a visualization of missing values"""
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    fig = px.bar(
        x=missing_pct.values,
        y=missing_pct.index,
        orientation='h',
        title='Missing Values by Column (%)',
        labels={'x': 'Percentage Missing (%)', 'y': 'Column'},
        color=missing_pct.values,
        color_continuous_scale='Reds'
    )
    fig.update_layout(height=max(400, len(missing_pct) * 20), showlegend=False)
    return fig

def plot_permit_status_distribution(df):
    """Plot distribution of permit statuses"""
    if 'Current Status' not in df.columns:
        return None

    status_counts = df['Current Status'].value_counts()

    fig = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title='Distribution of Permit Statuses',
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def plot_permits_over_time(df):
    """Plot permits filed over time"""
    if 'Filed Date' not in df.columns:
        return None

    df_time = df.copy()
    df_time['Year'] = df_time['Filed Date'].dt.year
    df_time['Month'] = df_time['Filed Date'].dt.to_period('M').astype(str)

    permits_by_month = df_time.groupby('Month').size().reset_index(name='Count')

    fig = px.line(
        permits_by_month,
        x='Month',
        y='Count',
        title='Number of Permits Filed Over Time',
        labels={'Month': 'Month', 'Count': 'Number of Permits'}
    )
    fig.update_xaxes(tickangle=45, nticks=20)
    fig.update_layout(height=500)
    return fig

def plot_permits_by_type(df):
    """Plot distribution of permit types"""
    if 'Permit Type Definition' not in df.columns:
        return None

    type_counts = df['Permit Type Definition'].value_counts().head(15)

    fig = px.bar(
        x=type_counts.values,
        y=type_counts.index,
        orientation='h',
        title='Top 15 Permit Types',
        labels={'x': 'Number of Permits', 'y': 'Permit Type'},
        color=type_counts.values,
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=500, showlegend=False)
    return fig

def plot_cost_distribution(df):
    """Plot distribution of estimated costs"""
    if 'Estimated Cost' not in df.columns:
        return None

    df_cost = df[df['Estimated Cost'].notna() & (df['Estimated Cost'] > 0)].copy()
    df_cost['Log Cost'] = np.log10(df_cost['Estimated Cost'] + 1)

    fig = px.histogram(
        df_cost,
        x='Log Cost',
        nbins=50,
        title='Distribution of Estimated Costs (Log Scale)',
        labels={'Log Cost': 'Log10(Estimated Cost)'},
        marginal='box'
    )
    fig.update_layout(height=500)
    return fig

def plot_neighborhood_analysis(df):
    """Analyze permits by neighborhood"""
    if 'Neighborhoods - Analysis Boundaries' not in df.columns:
        return None

    neighborhood_counts = df['Neighborhoods - Analysis Boundaries'].value_counts().head(20)

    fig = px.bar(
        x=neighborhood_counts.values,
        y=neighborhood_counts.index,
        orientation='h',
        title='Top 20 Neighborhoods by Number of Permits',
        labels={'x': 'Number of Permits', 'y': 'Neighborhood'},
        color=neighborhood_counts.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=600, showlegend=False)
    return fig

def plot_processing_time(df):
    """Analyze permit processing time"""
    if 'Filed Date' not in df.columns or 'Issued Date' not in df.columns:
        return None

    df_time = df[(df['Filed Date'].notna()) & (df['Issued Date'].notna())].copy()
    df_time['Processing Days'] = (df_time['Issued Date'] - df_time['Filed Date']).dt.days
    df_time = df_time[df_time['Processing Days'] >= 0]
    df_time = df_time[df_time['Processing Days'] <= df_time['Processing Days'].quantile(0.95)]

    fig = px.histogram(
        df_time,
        x='Processing Days',
        nbins=50,
        title='Distribution of Permit Processing Time (Days from Filed to Issued)',
        labels={'Processing Days': 'Days'},
        marginal='box'
    )
    fig.update_layout(height=500)
    return fig

def plot_construction_types(df):
    """Analyze proposed construction types"""
    if 'Proposed Construction Type Description' not in df.columns:
        return None

    const_types = df['Proposed Construction Type Description'].value_counts()

    fig = px.pie(
        values=const_types.values,
        names=const_types.index,
        title='Distribution of Proposed Construction Types',
        hole=0.3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def plot_stories_analysis(df):
    """Analyze building stories"""
    if 'Number of Proposed Stories' not in df.columns or 'Number of Existing Stories' not in df.columns:
        return None

    df_stories = df[(df['Number of Proposed Stories'].notna()) & 
                    (df['Number of Existing Stories'].notna())].copy()
    df_stories = df_stories[df_stories['Number of Proposed Stories'] <= 20]

    fig = px.scatter(
        df_stories,
        x='Number of Existing Stories',
        y='Number of Proposed Stories',
        title='Existing vs Proposed Number of Stories',
        labels={'Number of Existing Stories': 'Existing Stories', 
                'Number of Proposed Stories': 'Proposed Stories'},
        opacity=0.5
    )

    # Add diagonal line
    max_val = max(df_stories['Number of Existing Stories'].max(), 
                  df_stories['Number of Proposed Stories'].max())
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], 
                            mode='lines', name='No Change',
                            line=dict(color='red', dash='dash')))

    fig.update_layout(height=500)
    return fig

def plot_supervisor_district(df):
    """Analyze permits by supervisor district"""
    if 'Supervisor District' not in df.columns:
        return None

    district_counts = df['Supervisor District'].value_counts().sort_index()

    fig = px.bar(
        x=district_counts.index,
        y=district_counts.values,
        title='Number of Permits by Supervisor District',
        labels={'x': 'Supervisor District', 'y': 'Number of Permits'},
        color=district_counts.values,
        color_continuous_scale='Teal'
    )
    fig.update_layout(height=500, showlegend=False)
    return fig

# Main application
def main():
    # Header
    st.markdown('<p class="main-header">🏗️ San Francisco Building Permits Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive Dashboard for Building Permit Data Exploration</p>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")

    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload CleanedData.csv",
        type=['csv'],
        help="Upload the building permits CSV file"
    )

    # Load data
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        # Try to load from default location
        try:
            df = load_data("CleanedData.csv")
        except:
            df = None

    if df is None:
        st.info("👆 Please upload the CleanedData.csv file using the sidebar to begin analysis.")
        st.stop()

    # Calculate statistics
    stats = calculate_statistics(df)

    # Sidebar filters
    st.sidebar.markdown("### Filters")

    # Date range filter
    if 'Filed Date' in df.columns:
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(stats['date_range'][0], stats['date_range'][1]),
            min_value=stats['date_range'][0],
            max_value=stats['date_range'][1]
        )

        if len(date_range) == 2:
            df = df[(df['Filed Date'] >= pd.Timestamp(date_range[0])) & 
                   (df['Filed Date'] <= pd.Timestamp(date_range[1]))]

    # Status filter
    if 'Current Status' in df.columns:
        statuses = ['All'] + sorted(df['Current Status'].dropna().unique().tolist())
        selected_status = st.sidebar.selectbox("Permit Status", statuses)

        if selected_status != 'All':
            df = df[df['Current Status'] == selected_status]

    # Permit type filter
    if 'Permit Type Definition' in df.columns:
        permit_types = ['All'] + sorted(df['Permit Type Definition'].dropna().unique().tolist())
        selected_type = st.sidebar.selectbox("Permit Type", permit_types)

        if selected_type != 'All':
            df = df[df['Permit Type Definition'] == selected_type]

    st.sidebar.markdown("---")
    st.sidebar.info(f"Filtered records: {len(df):,}")

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Temporal Analysis",
        "🏘️ Geographical Analysis",
        "💰 Cost & Construction",
        "📋 Data Explorer"
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

        with col2:
            st.subheader("Top Permit Types")
            fig = plot_permits_by_type(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        # Missing values analysis
        st.markdown("---")
        st.subheader("Data Quality: Missing Values Analysis")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = plot_missing_values(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Key Insights")
            missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
            missing_pct = missing_pct[missing_pct > 50]

            st.markdown("**Columns with >50% missing:**")
            for col, pct in missing_pct.items():
                st.write(f"- {col}: {pct:.1f}%")

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
            if 'Filed Date' in df.columns:
                st.metric("Earliest Permit", df['Filed Date'].min().strftime('%Y-%m-%d'))
                st.metric("Latest Permit", df['Filed Date'].max().strftime('%Y-%m-%d'))

                # Average permits per year
                years = (df['Filed Date'].max() - df['Filed Date'].min()).days / 365.25
                avg_per_year = len(df) / years if years > 0 else 0
                st.metric("Avg Permits/Year", f"{avg_per_year:,.0f}")

        st.markdown("---")

        st.subheader("Permit Processing Time Analysis")
        fig = plot_processing_time(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

            # Processing time statistics
            if 'Filed Date' in df.columns and 'Issued Date' in df.columns:
                df_time = df[(df['Filed Date'].notna()) & (df['Issued Date'].notna())].copy()
                df_time['Processing Days'] = (df_time['Issued Date'] - df_time['Filed Date']).dt.days
                df_time = df_time[df_time['Processing Days'] >= 0]

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Median Processing", f"{df_time['Processing Days'].median():.0f} days")
                with col2:
                    st.metric("Mean Processing", f"{df_time['Processing Days'].mean():.0f} days")
                with col3:
                    st.metric("Min Processing", f"{df_time['Processing Days'].min():.0f} days")
                with col4:
                    st.metric("Max Processing", f"{df_time['Processing Days'].quantile(0.95):.0f} days (95%)")
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
                title='Top 15 Zipcodes by Number of Permits',
                labels={'x': 'Zipcode', 'y': 'Number of Permits'},
                color=zipcode_counts.values,
                color_continuous_scale='Greens'
            )
            fig.update_layout(height=500, showlegend=False)
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
        st.info(f"Showing {len(df):,} records after filtering")

        # Column selector
        all_columns = df.columns.tolist()
        default_columns = [col for col in ['Permit Number', 'Permit Type Definition', 
                                           'Current Status', 'Filed Date', 'Neighborhoods - Analysis Boundaries',
                                           'Estimated Cost'] if col in all_columns]

        selected_columns = st.multiselect(
            "Select columns to display",
            options=all_columns,
            default=default_columns
        )

        if selected_columns:
            st.dataframe(df[selected_columns].head(100), use_container_width=True)
        else:
            st.dataframe(df.head(100), use_container_width=True)

        # Download filtered data
        st.markdown("---")
        st.markdown("### Download Data")

        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"filtered_permits_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

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
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p>Building Permits Analysis Dashboard | San Francisco Open Data</p>
            <p>Built with Streamlit 🎈 | Data updated: {}</p>
        </div>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
