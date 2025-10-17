from flask import Flask, request, jsonify
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List
import numpy as np
import threading
import time

app = Flask(__name__)

# Store request timestamps
request_timestamps = []
request_lock = threading.Lock()

def calculate_rps_by_second(log_times: List[datetime]):
    """Calculate RPS per second"""
    if not log_times:
        return [], []
    
    log_times.sort()
    start_time = log_times[0]
    end_time = log_times[-1]
    
    total_seconds = int((end_time - start_time).total_seconds()) + 1
    
    rps_dict = defaultdict(int)
    for log_time in log_times:
        second_offset = int((log_time - start_time).total_seconds())
        rps_dict[second_offset] += 1
    
    times = list(range(total_seconds))
    rps_values = [rps_dict.get(i, 0) for i in times]
    
    return times, rps_values

def plot_rps_over_time(times: List[float], rps_values: List[int], 
                       start_time: datetime, save_path: str = 'server_rps_analysis.png'):
    """Plot RPS line chart over time"""
    
    plt.figure(figsize=(14, 8))
    
    # Plot RPS line chart
    plt.subplot(2, 1, 1)
    plt.plot(times, rps_values, linewidth=1.5, color='#2E86AB', alpha=0.8)
    plt.fill_between(times, rps_values, alpha=0.3, color='#2E86AB')
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('RPS (requests/sec)', fontsize=12)
    plt.title('Server Received RPS Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics
    avg_rps = np.mean(rps_values)
    max_rps = np.max(rps_values)
    min_rps = np.min(rps_values)
    std_rps = np.std(rps_values)
    
    stats_text = f'Avg RPS: {avg_rps:.2f}\nMax RPS: {max_rps}\nMin RPS: {min_rps}\nSTD: {std_rps:.2f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot RPS distribution histogram
    plt.subplot(2, 1, 2)
    plt.hist(rps_values, bins=30, color='#A23B72', alpha=0.7, edgecolor='black')
    plt.xlabel('RPS (requests/sec)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('RPS Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.axvline(avg_rps, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_rps:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nChart saved to: {save_path}")

def print_statistics(times: List[float], rps_values: List[int], log_times: List[datetime]):
    """Print detailed statistics"""
    if not rps_values:
        print("No data")
        return
    
    log_times.sort()
    start_time = log_times[0]
    end_time = log_times[-1]
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*60)
    print("Server Received RPS Statistics Analysis")
    print("="*60)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    print(f"Total duration: {duration:.2f} seconds")
    print(f"Total requests: {len(log_times)}")
    print(f"\nAverage RPS: {np.mean(rps_values):.2f}")
    print(f"Max RPS: {np.max(rps_values)}")
    print(f"Min RPS: {np.min(rps_values)}")
    print(f"Median RPS: {np.median(rps_values):.2f}")
    print(f"Standard Deviation: {np.std(rps_values):.2f}")
    print(f"Coefficient of Variation (CV): {np.std(rps_values)/np.mean(rps_values)*100:.2f}%")
    
    # Find peak RPS moment
    max_rps_idx = np.argmax(rps_values)
    max_rps_time = start_time.timestamp() + max_rps_idx
    max_rps_datetime = datetime.fromtimestamp(max_rps_time)
    print(f"\nRPS peak time: {max_rps_datetime.strftime('%Y-%m-%d %H:%M:%S')} "
          f"(offset +{max_rps_idx}s)")
    
    # RPS distribution percentiles
    print(f"\nRPS Distribution:")
    percentiles = [25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(rps_values, p)
        print(f"  P{p}: {value:.2f}")
    
    # Zero request period statistics
    zero_rps_count = sum(1 for rps in rps_values if rps == 0)
    zero_rps_percentage = (zero_rps_count / len(rps_values)) * 100
    print(f"\nZero request periods: {zero_rps_count} seconds ({zero_rps_percentage:.2f}%)")
    print("="*60)

def generate_statistics():
    """Generate statistics and charts"""
    with request_lock:
        if not request_timestamps:
            print("No request data to analyze")
            return
        
        timestamps = request_timestamps.copy()
    
    print(f"\nGenerating statistics for {len(timestamps)} requests...")
    
    # Calculate RPS per second
    times, rps_values = calculate_rps_by_second(timestamps)
    
    # Print statistics
    print_statistics(times, rps_values, timestamps)
    
    # Plot chart
    plot_rps_over_time(times, rps_values, timestamps[0])
    
    print("\nStatistics generation completed!")

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """Receive OpenAI format chat completion requests"""
    try:
        # Record request timestamp
        with request_lock:
            request_timestamps.append(datetime.now())
        
        # Get request data
        data = request.json
        
        # Simple response (mock OpenAI response)
        response = {
            "id": f"chatcmpl-{len(request_timestamps)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": data.get('model', 'unknown'),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a mock response from the test server."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20
            }
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"Error handling request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/v1/chat/completions/finish', methods=['POST'])
def finish():
    """Receive notification that all requests are sent, generate statistics"""
    try:
        data = request.json
        total_requests = data.get('total_requests', 0)
        
        print(f"\nReceived finish signal. Expected {total_requests} requests.")
        
        with request_lock:
            actual_requests = len(request_timestamps)
        
        print(f"Actually received {actual_requests} requests.")
        
        # Wait a moment to ensure all requests are received
        time.sleep(1)
        
        # Generate statistics in a separate thread
        threading.Thread(target=generate_statistics).start()
        
        return jsonify({"status": "Statistics generation started"}), 200
    
    except Exception as e:
        print(f"Error in finish handler: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get current statistics"""
    with request_lock:
        count = len(request_timestamps)
        if count > 0:
            start = request_timestamps[0]
            end = request_timestamps[-1]
            duration = (end - start).total_seconds()
            rps = count / duration if duration > 0 else 0
        else:
            start = end = None
            duration = 0
            rps = 0
    
    return jsonify({
        "total_requests": count,
        "start_time": start.isoformat() if start else None,
        "end_time": end.isoformat() if end else None,
        "duration": duration,
        "average_rps": rps
    })

@app.route('/reset', methods=['POST'])
def reset():
    """Reset all recorded data"""
    with request_lock:
        request_timestamps.clear()
    print("\nAll request data cleared")
    return jsonify({"status": "Reset successful"}), 200

if __name__ == '__main__':
    print("="*60)
    print("OpenAI Compatible Test Server")
    print("="*60)
    print("Endpoints:")
    print("  POST /v1/chat/completions - Receive chat requests")
    print("  POST /v1/chat/completions/finish - Generate statistics")
    print("  GET  /stats - Get current statistics")
    print("  POST /reset - Reset all data")
    print("="*60)
    print("\nServer starting on http://127.0.0.1:8000")
    print("Press Ctrl+C to stop\n")
    
    app.run(host='127.0.0.1', port=8000, debug=False)