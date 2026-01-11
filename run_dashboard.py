#!/usr/bin/env python
"""
Entry point for the Mission Control Dashboard.

This script provides a command-line interface to launch the Streamlit
dashboard for the High-Speed Railway Wagon Inspection System.

Usage:
    streamlit run run_dashboard.py -- [--video-source <path_or_url>]
    
Examples:
    # Run with default settings
    streamlit run run_dashboard.py
    
    # Run with a video file
    streamlit run run_dashboard.py -- --video-source path/to/video.mp4
    
    # Run with an RTSP stream
    streamlit run run_dashboard.py -- --video-source rtsp://camera.local/stream

Requirements: 1.1
"""

import argparse
import sys
import streamlit as st

from dashboard.app import MissionControlDashboard


def parse_args():
    """
    Parse command-line arguments for the dashboard.
    
    Returns:
        argparse.Namespace with parsed arguments:
            - video_source: Default video source (RTSP URL or file path)
    """
    parser = argparse.ArgumentParser(
        description="Mission Control Dashboard for Railway Wagon Inspection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    streamlit run run_dashboard.py
    streamlit run run_dashboard.py -- --video-source video.mp4
    streamlit run run_dashboard.py -- --video-source rtsp://camera/stream
        """
    )
    
    parser.add_argument(
        "--video-source",
        type=str,
        default="",
        help="Default video source (RTSP URL or file path)"
    )
    
    # Parse known args to handle Streamlit's own arguments
    args, _ = parser.parse_known_args()
    return args


def main():
    """
    Main entry point for the dashboard application.
    
    Parses command-line arguments, sets up default video source if provided,
    and launches the Mission Control Dashboard.
    
    Requirements: 1.1
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Set default video source in session state if provided via command line
    if args.video_source:
        if "video_source" not in st.session_state or not st.session_state.video_source:
            st.session_state.video_source = args.video_source
    
    # Create and run the dashboard
    dashboard = MissionControlDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
