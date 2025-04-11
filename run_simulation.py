import time
import subprocess
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def ensure_file_exists(filepath):
    """Make sure the file exists and is writable"""
    try:
        Path(filepath).touch(exist_ok=True)
        return os.path.exists(filepath)
    except:
        return False

def run_log_simulation(duration=10):
    """Run the log simulation for a fixed duration"""
    clear_screen()
    print("=" * 80)
    print(f"STEP 1: Starting log simulation for {duration} seconds...")
    print("=" * 80)
    
    # Make sure we're starting with a clean log file
    log_file = 'api_logs.json'
    if os.path.exists(log_file):
        print(f"Clearing existing log file: {log_file}")
        with open(log_file, 'w') as f:
            pass  # Just clear the file
    
    # Check if the simulation script exists
    sim_script = 'simulate_logs/simuate_logs.py'
    if not os.path.exists(sim_script):
        print(f"Error: Log simulation script not found: {sim_script}")
        return False
    
    # Run the simulation for the specified duration
    print("\nGenerating logs... Press Ctrl+C to stop early")
    sim_process = subprocess.Popen([sys.executable, sim_script])
    try:
        for i in range(duration):
            print(f"Simulating logs: {i+1}/{duration} seconds", end="\r")
            time.sleep(1)  # Let it run for the specified time
    except KeyboardInterrupt:
        print("\nLog simulation stopped by user.")
    finally:
        sim_process.terminate()  # Stop the simulation
        sim_process.wait()
    
    print("\nLog simulation complete.")
    
    # Check if logs were generated
    if os.path.exists(log_file):
        size = os.path.getsize(log_file)
        print(f"Log file size: {size} bytes")
        return size > 0
    return False

def run_anomaly_detection():
    """Run the file-based anomaly detection"""
    clear_screen()
    print("\n" + "=" * 80)
    print("STEP 2: Running file-based anomaly detection...")
    print("=" * 80 + "\n")
    
    try:
        from agents.file_based_anomaly_agent import FileBasedAnomalyAgent
        
        # Create agent
        agent = FileBasedAnomalyAgent()
        
        # Load logs from file
        logs = agent.load_logs_from_file('api_logs.json')
        
        if logs:
            # Detect anomalies
            anomalies, indices, scores = agent.detect_anomalies(logs, threshold=0.8)
            
            # Print results
            print(f"\nFound {len(anomalies)} anomalies from {len(logs)} logs")
            
            # Show top anomalies
            if anomalies:
                print("\nTop anomalies:")
                # Sort anomalies by score for display
                sorted_anomalies = sorted(anomalies, key=lambda x: x['score'], reverse=True)
                for i, anomaly in enumerate(sorted_anomalies[:5]):
                    print(f"Anomaly {i+1}:")
                    print(f"  Score: {anomaly['score']:.2f}")
                    print(f"  Value: {anomaly['value']:.2f} ms")
                    print(f"  API: {anomaly['context'].get('api_name')}")
                    print(f"  Environment: {anomaly['context'].get('environment')}")
                    print()
                
                print("\nSee 'anomaly_detection_results.png' for visualization.")
                return anomalies
            else:
                print("No anomalies detected with current threshold.")
                return []
        else:
            print("No logs available for analysis")
            return []
            
    except Exception as e:
        print(f"Error running anomaly detection: {str(e)}")
        return []
    
def run_crew_analysis():
    """Run the CrewAI analysis"""
    clear_screen()
    print("\n" + "=" * 80)
    print("STEP 3: Running CrewAI analysis for deeper insights...")
    print("=" * 80 + "\n")
    
    try:
        from file_based_crew import FileCrew
        
        # Create the FileCrew
        crew = FileCrew()
        
        # Run a monitoring cycle
        result = crew.run_monitoring_cycle('api_logs.json')
        
        return result
    except Exception as e:
        print(f"Error running CrewAI analysis: {str(e)}")
        return None

def main():
    clear_screen()
    print("=" * 80)
    print("API MONITORING SYSTEM: ANOMALY DETECTION & AGENT ANALYSIS")
    print("=" * 80)
    print("\nThis simulation will:")
    print("1. Generate simulated API logs")
    print("2. Detect anomalies using RRCF algorithm")
    print("3. Run CrewAI agents to analyze the anomalies and generate insights")
    
    input("\nPress Enter to begin simulation...")
    
    # Step 1: Run log simulation
    log_success = run_log_simulation(duration=20)
    
    if not log_success:
        print("Error: Failed to generate logs. Pipeline stopped.")
        return
    
    # Wait for user to continue
    input("\nPress Enter to continue to anomaly detection...")
    
    # Step 2: Run anomaly detection
    anomalies = run_anomaly_detection()
    
    if not anomalies:
        print("\nNo anomalies found for CrewAI to analyze.")
        choice = input("Continue with CrewAI analysis anyway? (y/n): ").lower()
        if choice != 'y':
            print("Simulation ended.")
            return
    
    # Wait for user to continue
    input("\nPress Enter to continue to CrewAI analysis...")
    
    # Step 3: Run CrewAI analysis
    run_crew_analysis()
    
    print("\n" + "=" * 80)
    print("Simulation and Analysis Pipeline Complete!")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user. Exiting...")