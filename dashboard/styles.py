"""
CSS injection module for the Mission Control Dashboard.

Provides custom dark-themed styling for a professional industrial monitoring interface.
"""

import streamlit as st

CUSTOM_CSS = """
<style>
    /* Reduce top padding for maximum screen utilization */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    
    /* Dark metric card styling */
    div[data-testid="metric-container"] {
        background-color: #1E1E1E;
        border: 1px solid #3D3D3D;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Large bold metric values */
    div[data-testid="metric-container"] > div > div > div {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Metric label styling */
    div[data-testid="metric-container"] label {
        font-size: 0.9rem;
        color: #B0B0B0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Status indicator colors */
    .status-normal {
        color: #00FF00;
        font-weight: bold;
    }
    
    .status-alert {
        color: #FF4444;
        font-weight: bold;
    }
    
    /* Video container styling */
    .video-container {
        border: 2px solid #3D3D3D;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Detection log styling */
    .detection-log {
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 10px;
    }
</style>
"""


def inject_css() -> None:
    """
    Inject custom CSS into the Streamlit app.
    
    Uses st.markdown with unsafe_allow_html=True to apply
    dark-themed styling throughout the dashboard.
    """
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
