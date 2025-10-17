import argparse
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import requests


def parse_log_time(log_time_str: str) -> datetime:
    """Parse log_time string to datetime object."""
    # 支持带毫秒的小数秒
    return datetime.strptime(log_time_str, "%Y-%m-%d %H:%M:%S.%f")


def load_requests(file_path: str) -> List[Dict]:
    """Load all requests from JSONL log file.

    期望每行结构类似：
    {
      "log_time": "2025-10-16 12:34:56.789",
      "request": {"request_json": {...}}
    }
    """
    requests_data: List[Dict] = []
    failed_lines = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                log_time = parse_log_time(data['log_time'])
                request_json = data['request']['request_json']
                requests_data.append({'log_time': log_time, 'request_json': request_json})
            except Exception as e:
                failed_lines += 1
                print(f"Failed to parse line: {e}")
    if failed_lines > 0:
        print(f"Warning: skipped {failed_lines} malformed line(s)")
    return requests_data


def calculate_original_rps(requests_data: List[Dict]) -> float:
    if len(requests_data) < 2:
        return 0.0
    requests_data.sort(key=lambda x: x['log_time'])
    time_span = (requests_data[-1]['log_time'] - requests_data[0]['log_time']).total_seconds()
    if time_span <= 0:
        return 0.0
    return (len(requests_data) - 1) / time_span


def send_requests_with_original_timing(
    requests_data: List[Dict],
    api_url: str,
    speed: float = 1.0,
    notify_finish_url: Optional[str] = None,
    timeout: float = 60.0,
    retries: int = 2,
    retry_backoff: float = 1.5,
    start_index: int = 0,
    max_requests: int = 0,
) -> None:
    """根据日志间隔回放，支持加速倍率 speed (>1 加速, <1 减速)。"""
    if not requests_data:
        print("No request data")
        return

    requests_data.sort(key=lambda x: x['log_time'])
    print(f"Total {len(requests_data)} requests")
    print(f"Time range: {requests_data[0]['log_time']} to {requests_data[-1]['log_time']}")
    if speed != 1.0:
        print(f"Speed factor: {speed}x")

    # 起始与数量裁剪
    total_len = len(requests_data)
    start_index = max(0, min(start_index, total_len - 1))
    end_index = total_len if max_requests <= 0 else min(total_len, start_index + max_requests)
    slice_data = requests_data[start_index:end_index]

    start_time = time.time()
    base_log_time = slice_data[0]['log_time']
    success_count = 0
    fail_count = 0

    for i, req_data in enumerate(slice_data):
        # 目标相对时间（应用速度倍率）
        target_rel = (req_data['log_time'] - base_log_time).total_seconds() / max(speed, 1e-9)
        elapsed = time.time() - start_time
        delay = max(0.0, target_rel - elapsed)
        if delay > 0:
            time.sleep(delay)

        ok, status, err_text = send_with_retry(api_url, req_data['request_json'], timeout, retries, retry_backoff)
        elapsed_now = time.time() - start_time
        if ok:
            success_count += 1
            print(f"[{elapsed_now:.3f}s] Request {start_index + i + 1}/{total_len} - {status}")
        else:
            fail_count += 1
            detail = f" (detail: {err_text[:200]})" if err_text else ""
            print(f"[{elapsed_now:.3f}s] Request {start_index + i + 1}/{total_len} - Failed{detail}")

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Sending completed!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Success: {success_count}, Failed: {fail_count}")
    if total_time > 0:
        print(f"Actual RPS: {len(requests_data)/total_time:.2f}")
    print("=" * 60)

    # 通知服务器统计
    if notify_finish_url:
        try:
            requests.post(notify_finish_url, json={"total_requests": len(requests_data)}, timeout=5)
            print("\nStatistics generation requested from server")
        except Exception as e:
            print(f"\nFailed to notify server: {e}")


def send_requests_with_fixed_rps(
    requests_data: List[Dict],
    api_url: str,
    rps: float,
    notify_finish_url: Optional[str] = None,
    timeout: float = 60.0,
    retries: int = 2,
    retry_backoff: float = 1.5,
    start_index: int = 0,
    max_requests: int = 0,
) -> None:
    if not requests_data:
        print("No request data")
        return

    rps = max(1e-6, rps)
    interval = 1.0 / rps
    requests_data.sort(key=lambda x: x['log_time'])
    print(f"Total {len(requests_data)} requests")
    print(f"RPS: {rps:.3f} (interval {interval:.3f}s)")

    # 起始与数量裁剪
    total_len = len(requests_data)
    start_index = max(0, min(start_index, total_len - 1))
    end_index = total_len if max_requests <= 0 else min(total_len, start_index + max_requests)
    slice_data = requests_data[start_index:end_index]

    start_time = time.time()
    success_count = 0
    fail_count = 0

    for i, req_data in enumerate(slice_data):
        target_time = start_time + i * interval
        delay = max(0.0, target_time - time.time())
        if delay > 0:
            time.sleep(delay)

        ok, status, err_text = send_with_retry(api_url, req_data['request_json'], timeout, retries, retry_backoff)
        elapsed_now = time.time() - start_time
        if ok:
            success_count += 1
            print(f"[{elapsed_now:.3f}s] Request {start_index + i + 1}/{total_len} - {status}")
        else:
            fail_count += 1
            detail = f" (detail: {err_text[:200]})" if err_text else ""
            print(f"[{elapsed_now:.3f}s] Request {start_index + i + 1}/{total_len} - Failed{detail}")

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Sending completed!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Success: {success_count}, Failed: {fail_count}")
    if total_time > 0:
        print(f"Actual RPS: {len(requests_data)/total_time:.2f}")
    print("=" * 60)

    if notify_finish_url:
        try:
            requests.post(notify_finish_url, json={"total_requests": len(requests_data)}, timeout=5)
            print("\nStatistics generation requested from server")
        except Exception as e:
            print(f"\nFailed to notify server: {e}")


def send_with_retry(api_url: str, payload: Dict, timeout: float, retries: int, backoff: float) -> Tuple[bool, int, str]:
    """发送请求并重试。返回 (ok, status_code, err_text)."""
    attempt = 0
    last_err = ""
    while attempt <= max(0, retries):
        try:
            resp = requests.post(api_url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                return True, resp.status_code, ""
            last_err = resp.text or f"HTTP {resp.status_code}"
        except Exception as e:
            last_err = str(e)

        if attempt == retries:
            break
        sleep_s = (backoff ** attempt)
        time.sleep(sleep_s)
        attempt += 1

    return False, 0, last_err


def check_health(health_url: str, timeout: float = 5.0) -> bool:
    try:
        r = requests.get(health_url, timeout=timeout)
        return r.ok
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay requests from log by original intervals or fixed RPS")
    parser.add_argument("--file", required=False, default="data/AIA_main_deepseek_v3_1000.log", help="Path to JSONL log file")
    parser.add_argument("--api", required=False, default="http://127.0.0.1:8000/v1/chat/completions", help="Target API URL")
    parser.add_argument("--mode", choices=["interval", "rps"], default="interval", help="Replay mode: interval(original gaps) or rps")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed factor for interval mode (>1 faster)")
    parser.add_argument("--rps", type=float, default=0.0, help="Fixed RPS when mode=rps; default uses original average RPS")
    parser.add_argument("--notify", default="", help="Optional notify URL, e.g., http://127.0.0.1:8000/v1/chat/completions/finish")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds")
    parser.add_argument("--retries", type=int, default=2, help="Retry times per request on failure")
    parser.add_argument("--retry-backoff", type=float, default=1.5, help="Exponential backoff base for retries")
    parser.add_argument("--start", type=int, default=0, help="Start index in dataset (0-based)")
    parser.add_argument("--max", type=int, default=0, help="Max requests to send (0 means all)")
    parser.add_argument("--health", default="http://127.0.0.1:8000/health", help="Health check URL (empty to skip)")

    args = parser.parse_args()

    print("Loading request data...")
    reqs = load_requests(args.file)
    if not reqs:
        print("No valid request data found")
        return

    original_rps = calculate_original_rps(reqs)
    print(f"Original data average RPS: {original_rps:.3f}")

    notify_url = args.notify if args.notify else None

    if args.health:
        ok = check_health(args.health)
        if not ok:
            print(f"Warning: health check failed for {args.health}. The server may be unavailable or cold-starting.")

    if args.mode == "interval":
        print("\nStarting interval-based replay...\n")
        send_requests_with_original_timing(
            reqs,
            api_url=args.api,
            speed=args.speed,
            notify_finish_url=notify_url,
            timeout=args.timeout,
            retries=args.retries,
            retry_backoff=args.retry_backoff,
            start_index=args.start,
            max_requests=args.max,
        )
    else:
        rps_value = args.rps if args.rps > 0 else (original_rps if original_rps > 0 else 1.0)
        print(f"\nStarting fixed-RPS replay at {rps_value:.3f}...\n")
        send_requests_with_fixed_rps(
            reqs,
            api_url=args.api,
            rps=rps_value,
            notify_finish_url=notify_url,
            timeout=args.timeout,
            retries=args.retries,
            retry_backoff=args.retry_backoff,
            start_index=args.start,
            max_requests=args.max,
        )


if __name__ == "__main__":
    main()


