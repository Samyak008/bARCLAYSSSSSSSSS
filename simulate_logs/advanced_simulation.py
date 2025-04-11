import time
import random
import json
import math
import numpy as np
from datetime import datetime, timedelta
import uuid

# Constants
api_names = ["frontend", "inventory", "payment", "authentication", "checkout", "shipping"]
environments = ["cloud_a", "on_prem", "cloud_b", "edge", "hybrid"]
log_file = "api_logs.json"
base_response_times = {
    "frontend": 200,
    "inventory": 150,
    "payment": 350,
    "authentication": 100,
    "checkout": 300,
    "shipping": 250
}
dependency_map = {
    "frontend": ["authentication", "inventory"],
    "checkout": ["inventory", "payment", "shipping"],
    "payment": ["authentication"],
    "shipping": ["inventory"]
}

class PatternGenerator:
    def __init__(self):
        # Start with a random time to get different time-of-day effects
        self.current_time = datetime.now() - timedelta(hours=random.randint(0, 23))
        self.correlation_ids = {}  # Store active correlation IDs
        self.anomaly_windows = []  # Time windows where anomalies occur
        self.degradation_services = {}  # Services experiencing gradual degradation
        self.cyclical_phase = random.random() * 2 * math.pi  # Random starting phase
    
    def set_anomaly_windows(self, count=3, duration_seconds=30):
        """Define time windows where anomalies will be more likely"""
        self.anomaly_windows = []
        for _ in range(count):
            start = self.current_time + timedelta(seconds=random.randint(30, 300))
            end = start + timedelta(seconds=duration_seconds)
            affected_service = random.choice(api_names)
            self.anomaly_windows.append({
                'start': start,
                'end': end,
                'service': affected_service,
                'magnitude': random.uniform(2.5, 5.0)  # How much to multiply response times
            })
    
    def set_degrading_services(self, count=2):
        """Define services that will experience gradual degradation"""
        services = random.sample(api_names, count)
        for service in services:
            start_after = random.randint(60, 180)  # Start degrading after X seconds
            degradation_rate = random.uniform(0.01, 0.04)  # % increase per second
            self.degradation_services[service] = {
                'start': self.current_time + timedelta(seconds=start_after),
                'rate': degradation_rate,
                'current_factor': 1.0
            }
    
    def get_cyclical_factor(self):
        """Return a cyclical factor based on time of day"""
        # Get the hour in a cyclical form (0-23)
        hour = self.current_time.hour
        # Convert to radians (0-2π)
        hour_rad = (hour / 24.0) * 2 * math.pi
        # Get a sinusoidal factor with a random phase
        return 1.0 + 0.3 * math.sin(hour_rad + self.cyclical_phase)
    
    def is_in_anomaly_window(self, service):
        """Check if the current time is in an anomaly window for the service"""
        now = self.current_time
        for window in self.anomaly_windows:
            if window['start'] <= now <= window['end'] and (window['service'] == service or window['service'] == 'all'):
                return window['magnitude']
        return 1.0  # No anomaly
    
    def get_degradation_factor(self, service):
        """Get the current degradation factor for a service"""
        if service in self.degradation_services:
            deg_info = self.degradation_services[service]
            if self.current_time > deg_info['start']:
                elapsed = (self.current_time - deg_info['start']).total_seconds()
                # Update the factor
                deg_info['current_factor'] = 1.0 + (elapsed * deg_info['rate'])
                return deg_info['current_factor']
        return 1.0  # No degradation
    
    def generate_correlated_request(self, parent_correlation_id=None):
        """Generate a correlated request following parent's characteristics"""
        timestamp = self.current_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        
        # Create or use correlation ID
        if parent_correlation_id:
            correlation_id = parent_correlation_id
        else:
            correlation_id = f"cid-{uuid.uuid4()}"
            self.correlation_ids[correlation_id] = {
                'created': self.current_time,
                'services_called': []
            }
        
        # Determine which API to call
        if parent_correlation_id:
            # This is a follow-up call in a chain
            parent_info = self.correlation_ids.get(parent_correlation_id)
            if parent_info:
                last_service = parent_info['services_called'][-1]
                possible_next = dependency_map.get(last_service, [])
                if possible_next:
                    api_name = random.choice(possible_next)
                else:
                    api_name = random.choice(api_names)
                parent_info['services_called'].append(api_name)
        else:
            # This is a new request chain, usually starting with frontend or authentication
            api_name = random.choice(["frontend", "authentication"])
            self.correlation_ids[correlation_id]['services_called'].append(api_name)
        
        # Choose environment - in real systems, often related services run in the same environment
        if parent_correlation_id and len(self.correlation_ids[correlation_id]['services_called']) > 1:
            # 80% chance to use the same environment as the previous call in the chain
            prev_env = self.correlation_ids[correlation_id].get('last_env')
            if prev_env and random.random() < 0.8:
                environment = prev_env
            else:
                environment = random.choice(environments)
                self.correlation_ids[correlation_id]['last_env'] = environment
        else:
            environment = random.choice(environments)
            if correlation_id in self.correlation_ids:
                self.correlation_ids[correlation_id]['last_env'] = environment
        
        # Calculate response time with various factors
        base_time = base_response_times.get(api_name, 200)
        
        # Apply factors
        anomaly_factor = self.is_in_anomaly_window(api_name)
        degradation_factor = self.get_degradation_factor(api_name)
        cyclical_factor = self.get_cyclical_factor()
        random_factor = random.normalvariate(1.0, 0.15)  # Normal distribution around 1.0
        
        # Combine all factors
        response_time = base_time * anomaly_factor * degradation_factor * cyclical_factor * random_factor
        
        # Randomly create errors, more likely during anomalies
        error_probability = 0.01  # Base 1% error rate
        if anomaly_factor > 1.0:
            error_probability *= anomaly_factor  # More errors during anomalies
        
        status_code = 200
        if random.random() < error_probability:
            status_codes = [400, 404, 500, 503]
            weights = [0.2, 0.2, 0.4, 0.2]
            status_code = random.choices(status_codes, weights=weights)[0]
        
        # Create log entry
        log_data = {
            "@timestamp": timestamp,
            "api_name": api_name,
            "response_time": response_time,
            "status_code": status_code,
            "correlation_id": correlation_id,
            "environment": environment,
            "message": f"API request to {api_name} completed in {response_time:.2f}ms with status {status_code}"
        }
        
        # Clean up old correlation IDs (older than 5 minutes)
        to_delete = []
        for cid, info in self.correlation_ids.items():
            if (self.current_time - info['created']).total_seconds() > 300:
                to_delete.append(cid)
        for cid in to_delete:
            del self.correlation_ids[cid]
        
        return log_data, correlation_id
    
    def advance_time(self, seconds=0.1):
        """Advance the simulation clock"""
        self.current_time += timedelta(seconds=seconds)

def main():
    print("Starting advanced log generation...")
    print(f"Writing logs to {log_file}")
    
    # Initialize the pattern generator
    generator = PatternGenerator()
    
    # Set up patterns
    generator.set_anomaly_windows(count=4, duration_seconds=60)
    generator.set_degrading_services(count=2)
    
    count = 0
    try:
        while True:
            # 30% chance to continue an existing request chain
            active_cids = list(generator.correlation_ids.keys())
            if active_cids and random.random() < 0.3:
                parent_id = random.choice(active_cids)
                log_data, _ = generator.generate_correlated_request(parent_id)
            else:
                # Start a new request chain
                log_data, _ = generator.generate_correlated_request()
            
            # Write to file
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_data) + "\n")
            
            count += 1
            if count % 10 == 0:
                print(f"Generated {count} logs... Latest: {log_data['api_name']} - {log_data['response_time']:.2f}ms")
                
            # Advance time
            time_increment = random.uniform(0.1, 0.5)
            generator.advance_time(time_increment)
            time.sleep(random.uniform(0.05, 0.2))  # Faster for testing
    
    except KeyboardInterrupt:
        print(f"\nLog generation stopped. Generated {count} logs.")

if __name__ == "__main__":
    main()