"""
Launch MLflow Tracking UI
=========================

Run this to view all experiments in your browser:
    python mlops/launch_ui.py

Then open: http://localhost:5000
"""

import os
import sys

if __name__ == "__main__":
    tracking_uri = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mlruns")
    print(f"Launching MLflow UI at: {tracking_uri}")
    print("Open your browser to: http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    
    os.system(f"mlflow ui --backend-store-uri file:///{tracking_uri} --port 5000")
