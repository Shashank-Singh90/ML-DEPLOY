#!/usr/bin/env python3
"""
Start MLflow UI for IoT Threat Detection experiments
"""
import subprocess
import sys
import os
import webbrowser
import time
from pathlib import Path

def start_mlflow_ui():
    """Start MLflow UI server"""
    
    # Ensure mlruns directory exists
    Path("mlruns").mkdir(exist_ok=True)
    
    print("🚀 Starting MLflow UI...")
    print("📊 Experiment tracking dashboard will be available at: http://localhost:5001")
    print("⏳ Starting server...")
    
    try:
        # Start MLflow UI
        process = subprocess.Popen([
            sys.executable, "-m", "mlflow", "ui",
            "--host", "0.0.0.0",
            "--port", "5001",
            "--backend-store-uri", "file:./mlruns"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        print("✅ MLflow UI started successfully!")
        print("🌐 Open your browser and go to: http://localhost:5001")
        print("📈 You can view experiments, compare models, and track metrics")
        print("🔄 To stop the server, press Ctrl+C")
        
        # Optionally open browser automatically
        try:
            webbrowser.open("http://localhost:5001")
            print("🌐 Browser opened automatically")
        except:
            print("🌐 Please manually open http://localhost:5001 in your browser")
        
        # Wait for the process to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping MLflow UI...")
        process.terminate()
        print("✅ MLflow UI stopped")
    except Exception as e:
        print(f"❌ Error starting MLflow UI: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    start_mlflow_ui()
