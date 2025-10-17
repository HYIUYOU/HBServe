import json
import time
from datetime import datetime
import requests
from typing import List, Dict

def parse_log_time(log_time_str: str) -> datetime:
    """Parse log_time string to datetime object"""
    return datetime.strptime(log_time_str, "%Y-%m-%d %H:%M:%S.%f")

def load_requests(file_path: str) -> List[Dict]:
    """Load all requests from JSON file"""
    requests_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    log_time = parse_log_time(data['log_time'])
                    request_json = data['request']['request_json']
                    requests_data.append({
                        'log_time': log_time,
                        'request_json': request_json
                    })
                except Exception as e:
                    print(f"Failed to parse line: {e}")
    return requests_data

def send_requests_with_original_timing(requests_data: List[Dict], api_url: str):
    """Send requests with original time intervals"""
    if not requests_data:
        print("No request data")
        return
    
    # Sort by log_time
    requests_data.sort(key=lambda x: x['log_time'])
    
    print(f"Total {len(requests_data)} requests")
    print(f"Time range: {requests_data[0]['log_time']} to {requests_data[-1]['log_time']}")
    
    start_time = time.time()
    success_count = 0
    fail_count = 0
    
    for i, req_data in enumerate(requests_data):
        # Calculate time difference from first request
        if i == 0:
            delay = 0
        else:
            time_diff = (req_data['log_time'] - requests_data[0]['log_time']).total_seconds()
            elapsed = time.time() - start_time
            delay = max(0, time_diff - elapsed)
        
        # Wait until send time
        if delay > 0:
            time.sleep(delay)
        
        # Send request
        try:
            response = requests.post(api_url, json=req_data['request_json'], timeout=30)
            elapsed = time.time() - start_time
            if response.status_code == 200:
                success_count += 1
                print(f"[{elapsed:.3f}s] Request {i+1}/{len(requests_data)} - Status: {response.status_code}")
            else:
                fail_count += 1
                print(f"[{elapsed:.3f}s] Request {i+1}/{len(requests_data)} - Status: {response.status_code} (Failed)")
        except Exception as e:
            elapsed = time.time() - start_time
            fail_count += 1
            print(f"[{elapsed:.3f}s] Request {i+1} failed: {e}")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Sending completed!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Success: {success_count}, Failed: {fail_count}")
    print(f"Actual RPS: {len(requests_data)/total_time:.2f}")
    print(f"{'='*60}")
    
    # Notify server to generate statistics
    try:
        requests.post(f"{api_url}/finish", json={"total_requests": len(requests_data)}, timeout=5)
        print("\nStatistics generation requested from server")
    except Exception as e:
        print(f"\nFailed to notify server: {e}")

def send_requests_with_fixed_rps(requests_data: List[Dict], api_url: str, rps: float = 1.0):
    """Send requests with fixed RPS"""
    if not requests_data:
        print("No request data")
        return
    
    # Sort by log_time
    requests_data.sort(key=lambda x: x['log_time'])
    
    print(f"Total {len(requests_data)} requests")
    print(f"RPS: {rps} (interval per request: {1/rps:.3f} seconds)")
    
    interval = 1.0 / rps
    start_time = time.time()
    success_count = 0
    fail_count = 0
    
    for i, req_data in enumerate(requests_data):
        # Calculate target send time
        target_time = start_time + i * interval
        current_time = time.time()
        delay = max(0, target_time - current_time)
        
        # Wait until send time
        if delay > 0:
            time.sleep(delay)
        
        # Send request
        try:
            response = requests.post(api_url, json=req_data['request_json'], timeout=30)
            elapsed = time.time() - start_time
            if response.status_code == 200:
                success_count += 1
                print(f"[{elapsed:.3f}s] Request {i+1}/{len(requests_data)} - Status: {response.status_code}")
            else:
                fail_count += 1
                print(f"[{elapsed:.3f}s] Request {i+1}/{len(requests_data)} - Status: {response.status_code} (Failed)")
        except Exception as e:
            elapsed = time.time() - start_time
            fail_count += 1
            print(f"[{elapsed:.3f}s] Request {i+1} failed: {e}")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Sending completed!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Success: {success_count}, Failed: {fail_count}")
    print(f"Actual RPS: {len(requests_data)/total_time:.2f}")
    print(f"{'='*60}")
    
    # Notify server to generate statistics
    try:
        requests.post(f"{api_url}/finish", json={"total_requests": len(requests_data)}, timeout=5)
        print("\nStatistics generation requested from server")
    except Exception as e:
        print(f"\nFailed to notify server: {e}")

def calculate_original_rps(requests_data: List[Dict]) -> float:
    """Calculate average RPS of original data"""
    if len(requests_data) < 2:
        return 0
    
    requests_data.sort(key=lambda x: x['log_time'])
    time_span = (requests_data[-1]['log_time'] - requests_data[0]['log_time']).total_seconds()
    
    if time_span > 0:
        return (len(requests_data) - 1) / time_span
    return 0

if __name__ == "__main__":
    # Configuration
    file_path = "./AIA_main_deepseek_v3_1000.log"
    api_url = "http://127.0.0.1:8000/v1/chat/completions"  # Local server endpoint
    
    # Load request data
    print("Loading request data...")
    requests_data = load_requests(file_path)
    
    if not requests_data:
        print("No valid request data found")
        exit(1)
    
    # Calculate original RPS
    original_rps = calculate_original_rps(requests_data)
    print(f"\nOriginal data average RPS: {original_rps:.3f}")
    
    # Select sending mode
    print("\nSelect sending mode:")
    print("1. Keep original time intervals")
    print("2. Use fixed RPS")
    
    mode = input("Please select (1/2): ").strip()
    
    if mode == "1":
        print("\nStarting to send requests with original time intervals...\n")
        send_requests_with_original_timing(requests_data, api_url)
    elif mode == "2":
        rps_input = input(f"Enter RPS (default {original_rps:.2f}): ").strip()
        rps = float(rps_input) if rps_input else original_rps
        print(f"\nStarting to send requests with fixed RPS ({rps})...\n")
        send_requests_with_fixed_rps(requests_data, api_url, rps)
    else:
        print("Invalid selection")