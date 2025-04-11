import time
import random
from datetime import datetime

api_names = ["frontend", "inventory", "payment"]
environments = ["cloud_a", "on_prem", "cloud_b"]
log_file = "api_logs.log"

def generate_log():
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    api_name = random.choice(api_names)
    response_time = round(random.uniform(0.1, 2.0), 1)
    status_code = 200 if random.random() > 0.05 else 500
    correlation_id = f"cid:{random.randint(10000, 99999)}"
    environment = random.choice(environments)
    log_entry = f"{timestamp} {api_name} {response_time} {status_code} {correlation_id} environment:{environment}\n"
    return log_entry

def main():
    with open(log_file, 'a') as f:
        while True:
            log = generate_log()
            f.write(log)
            print(f"Generated: {log.strip()}")
            time.sleep(1)

if __name__ == "__main__":
    main()
