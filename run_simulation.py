import time
import subprocess
import sys
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Disable telemetry and OpenTelemetry
os.environ["TELEMETRY_ENABLED"] = "false"
os.environ["OPENAI_TELEMETRY"] = "false" 
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Suppress error logs from telemetry
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
logging.getLogger("litellm").setLevel(logging.ERROR)

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

def run_log_simulation(duration=30, advanced=True):
    """Run the log simulation for a fixed duration"""
    clear_screen()
    print("=" * 80)
    print(f"STEP 1: Starting {'advanced' if advanced else 'standard'} log simulation for {duration} seconds...")
    print("=" * 80)
    
    # Make sure we're starting with a clean log file
    log_file = 'api_logs.json'
    if os.path.exists(log_file):
        print(f"Clearing existing log file: {log_file}")
        with open(log_file, 'w') as f:
            pass  # Just clear the file
    
    # Choose the simulation script
    if advanced:
        sim_script = 'simulate_logs/advanced_simulation.py'
        if not os.path.exists(sim_script):
            print(f"Advanced simulation script not found: {sim_script}, falling back to standard")
            sim_script = 'simulate_logs/simuate_logs.py'
    else:
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

def run_anomaly_detection(enhanced=True):
    """Run the anomaly detection"""
    clear_screen()
    print("\n" + "=" * 80)
    print(f"STEP 2: Running {'enhanced' if enhanced else 'standard'} anomaly detection...")
    print("=" * 80 + "\n")
    
    try:
        if enhanced:
            # Try to use enhanced agent
            try:
                from agents.enhanced_anomaly_agent import EnhancedAnomalyAgent
                agent = EnhancedAnomalyAgent()
                print("Using enhanced anomaly detection algorithm")
            except ImportError:
                print("Enhanced anomaly agent not found, falling back to standard")
                from agents.file_based_anomaly_agent import FileBasedAnomalyAgent
                agent = FileBasedAnomalyAgent()
        else:
            # Use standard agent
            from agents.file_based_anomaly_agent import FileBasedAnomalyAgent
            agent = FileBasedAnomalyAgent()
        
        # Load logs from file
        logs = agent.load_logs_from_file('api_logs.json')
        
        if logs:
            # Detect anomalies
            if enhanced:
                anomalies, indices, scores = agent.detect_anomalies(
                    logs, 
                    threshold=0.8, 
                    use_multidimensional=True, 
                    contextual_analysis=True
                )
            else:
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
                    print(f"  Status Code: {anomaly['context'].get('status_code', 'unknown')}")
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
        import traceback
        traceback.print_exc()
        return []

def run_forecasting(anomalies=None):
    """Run enhanced forecasting with anomaly integration"""
    clear_screen()
    print("\n" + "=" * 80)
    print("STEP 3: Running enhanced forecasting with anomaly integration...")
    print("=" * 80 + "\n")
    
    try:
        # Try to use enhanced forecasting
        try:
            from agents.enhanced_forecasting import EnhancedForecastingAgent
            agent = EnhancedForecastingAgent()
            print("Using enhanced forecasting algorithm")
        except ImportError:
            print("Enhanced forecasting agent not found, falling back to standard forecasting")
            return None
        
        # Load logs
        logs = agent.load_logs_from_file('api_logs.json')
        
        if logs:
            # Generate forecasts
            forecasts = agent.generate_forecasts(logs, anomalies)
            
            # Print results
            print(f"\nGenerated forecasts for {len(forecasts)} APIs")
            
            for api, forecast in list(forecasts.items())[:3]:  # Show first 3
                print(f"\n{api}:")
                print(f"  Current Average: {forecast['current_avg']:.2f}ms")
                print(f"  Forecast Average: {forecast['forecast_avg']:.2f}ms")
                print(f"  Trend: {forecast['trend']}")
                print(f"  Risk Level: {forecast.get('risk_level', 'unknown')}")
            
            print("\nSee 'forecast_predictions.png' for visualization.")
            return forecasts
        else:
            print("No logs available for forecasting")
            return None
    except Exception as e:
        print(f"Error running forecasting: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run_crew_analysis(anomalies, forecasts):
    """Run the CrewAI analysis with enhanced data"""
    clear_screen()
    print("\n" + "=" * 80)
    print("STEP 4: Running CrewAI analysis for deeper insights...")
    print("=" * 80 + "\n")
    
    try:
        from file_based_crew import FileCrew
        
        # Create the FileCrew
        crew = FileCrew()
        
        # Run a monitoring cycle with anomalies and forecasts
        result = crew.run_enhanced_monitoring_cycle('api_logs.json', anomalies, forecasts)
        return result
        # Clean up potential duplicate content in the result
        # if result:
        #     # Look for repeated sections and clean them up
        #     result_lines = result.splitlines()
        #     cleaned_lines = []
        #     seen_chunks = set()
            
        #     for line in result_lines:
        #         # Use a sliding window approach to detect significant duplicates
        #         if len(line.strip()) > 20:  # Only check substantial lines
        #             if line not in seen_chunks:
        #                 seen_chunks.add(line)
        #                 cleaned_lines.append(line)
        #         else:
        #             cleaned_lines.append(line)
            
        #     result = '\n'.join(cleaned_lines)
        
        # return result
    except Exception as e:
        print(f"Error running CrewAI analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    clear_screen()
    print("=" * 80)
    print("ENHANCED API MONITORING SYSTEM: ANOMALY DETECTION & FORECASTING")
    print("=" * 80)
    print("\nThis enhanced simulation will:")
    print("1. Generate advanced simulated API logs with patterns and anomalies")
    print("2. Detect anomalies using enhanced RRCF algorithm")
    print("3. Forecast future trends using time-series models")
    print("4. Run CrewAI agents to analyze and recommend actions")
    
    # Ask about simulation options
    print("\nSimulation options:")
    use_advanced = input("Use advanced log patterns (y/n)? [y]: ").lower() != 'n'
    duration = int(input("Duration for log generation in seconds [30]: ") or 30)
    use_enhanced = input("Use enhanced anomaly detection (y/n)? [y]: ").lower() != 'n'
    
    input("\nPress Enter to begin simulation...")
    
    # Step 1: Run log simulation
    log_success = run_log_simulation(duration=duration, advanced=use_advanced)
    
    if not log_success:
        print("Error: Failed to generate logs. Pipeline stopped.")
        return
    
    # Wait for user to continue
    input("\nPress Enter to continue to anomaly detection...")
    
    # Step 2: Run anomaly detection
    anomalies = run_anomaly_detection(enhanced=use_enhanced)
    
    # Wait for user to continue
    input("\nPress Enter to continue to forecasting...")
    
    # Step 3: Run forecasting
    forecasts = run_forecasting(anomalies)
    
    if not anomalies and not forecasts:
        print("\nNo data for CrewAI to analyze.")
        choice = input("Continue with CrewAI analysis anyway? (y/n): ").lower()
        if choice != 'y':
            print("Simulation ended.")
            return
    
    # Wait for user to continue
    input("\nPress Enter to continue to CrewAI analysis...")
    
    # Step 4: Run CrewAI analysis
    run_crew_analysis(anomalies, forecasts)
    
    print("\n" + "=" * 80)
    print("Enhanced Simulation and Analysis Pipeline Complete!")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user. Exiting...")