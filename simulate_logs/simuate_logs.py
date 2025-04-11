import time
import random
import json
from datetime import datetime

api_names = ["frontend", "inventory", "payment"]
environments = ["cloud_a", "on_prem", "cloud_b"]
log_file = "api_logs.json"  # Changed to JSON for easier parsing

def generate_log():
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    api_name = random.choice(api_names)
    response_time = round(random.uniform(0.1, 2.0), 3) * 1000  # Convert to ms
    
    # Occasionally generate anomalous response times
    if random.random() < 0.02:  # 2% chance of anomaly
        response_time = round(random.uniform(1.5, 5.0), 3) * 1000
    
    status_code = 200
    # Generate some errors
    if random.random() < 0.05:  # 5% error rate
        status_codes = [400, 404, 500, 503]
        weights = [0.2, 0.2, 0.4, 0.2]
        status_code = random.choices(status_codes, weights=weights)[0]
    
    correlation_id = f"cid:{random.randint(10000, 99999)}"
    environment = random.choice(environments)
    
    # Create a structured log entry
    log_data = {
        "@timestamp": timestamp,
        "api_name": api_name,
        "response_time": response_time,
        "status_code": status_code,
        "correlation_id": correlation_id,
        "environment": environment,
        "message": f"API request to {api_name} completed in {response_time}ms with status {status_code}"
    }
    
    return log_data

def main():
    print("Starting log generation...")
    print(f"Writing logs to {log_file}")
    
    count = 0
    try:
        while True:
            log_data = generate_log()
            
            # Write to file
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_data) + "\n")
            
            count += 1
            if count % 10 == 0:
                print(f"Generated {count} logs... Latest: {log_data['api_name']} - {log_data['response_time']}ms")
                
            time.sleep(random.uniform(0.1, 0.3))  # Faster for testing
    except KeyboardInterrupt:
        print(f"\nLog generation stopped. Generated {count} logs.")

if __name__ == "__main__":
    main()