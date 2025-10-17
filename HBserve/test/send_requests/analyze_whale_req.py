import json
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
from typing import List, Tuple
import numpy as np

def parse_log_time(log_time_str: str) -> datetime:
    """Parse log_time string to datetime object"""
    return datetime.strptime(log_time_str, "%Y-%m-%d %H:%M:%S.%f")

def load_log_times(file_path: str) -> List[datetime]:
    """Load all request timestamps from file"""
    log_times = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    log_time = parse_log_time(data['log_time'])
                    log_times.append(log_time)
                except Exception as e:
                    print(f"Failed to parse line: {e}")
    return log_times

def calculate_rps_by_second(log_times: List[datetime]) -> Tuple[List[float], List[int]]:
    """Calculate RPS per second
    
    Returns:
        times: List of seconds relative to start time
        rps_values: List of request counts per second
    """
    if not log_times:
        return [], []
    
    # Sort timestamps
    log_times.sort()
    
    # Start time
    start_time = log_times[0]
    end_time = log_times[-1]
    
    # Calculate total duration in seconds
    total_seconds = int((end_time - start_time).total_seconds()) + 1
    
    # Count requests per second
    rps_dict = defaultdict(int)
    for log_time in log_times:
        second_offset = int((log_time - start_time).total_seconds())
        rps_dict[second_offset] += 1
    
    # Generate continuous time series (fill with 0)
    times = list(range(total_seconds))
    rps_values = [rps_dict.get(i, 0) for i in times]
    
    return times, rps_values

def plot_rps_over_time(times: List[float], rps_values: List[int], 
                       start_time: datetime, save_path: str = 'rps_analysis.png'):
    """Plot RPS line chart over time"""
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot RPS line chart
    plt.subplot(2, 1, 1)
    plt.plot(times, rps_values, linewidth=1.5, color='#2E86AB', alpha=0.8)
    plt.fill_between(times, rps_values, alpha=0.3, color='#2E86AB')
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('RPS (requests/sec)', fontsize=12)
    plt.title('RPS Over Time', fontsize=14, fontweight='bold')
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
    plt.show()

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
    print("RPS Statistics Analysis")
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

def analyze_rps(file_path: str, save_path: str = 'rps_analysis.png'):
    """Main function: analyze RPS fluctuations from file"""
    print(f"Analyzing file: {file_path}")
    
    # Load timestamps
    log_times = load_log_times(file_path)
    
    if not log_times:
        print("No valid timestamp data found")
        return
    
    print(f"Successfully loaded {len(log_times)} request records")
    
    # Calculate RPS per second
    times, rps_values = calculate_rps_by_second(log_times)
    
    # Print statistics
    print_statistics(times, rps_values, log_times)
    
    # Plot chart
    plot_rps_over_time(times, rps_values, log_times[0], save_path)

if __name__ == "__main__":
    # Configuration
    file_path = "./AIA_main_deepseek_v3_1000.log"
    save_path = "rps_analysis.png"
    
    # Analyze RPS
    analyze_rps(file_path, save_path)