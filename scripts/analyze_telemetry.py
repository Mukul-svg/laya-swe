import os
import json
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
telemetry_path = os.path.join(PROJECT_ROOT, "benchmark_telemetry.jsonl")
records = []
if os.path.exists(telemetry_path):
    with open(telemetry_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
else:
    print(f"Warning: Telemetry log '{telemetry_path}' not found.")

if not records:
    print("No telemetry records found.")
    exit(0)

print("\nTurn-by-turn details:")
print(f"{'Turn':<5} | {'Task':<24} | {'Mode':<18} | {'Raw':<8} | {'Sent':<8} | {'Saved':<8} | {'Pct':<7} | {'Lat'}")
print("-" * 90)
for idx, r in enumerate(records):
    raw = r.get("raw_prompt_tokens", 0)
    sent = r.get("sent_prompt_tokens", 0)
    saved = r.get("tokens_saved", 0)
    pct = r.get("savings_percentage", 0.0)
    lat = r.get("latency_seconds", 0.0)
    print(f"{idx+1:<5} | {r.get('task_id',''):<24} | {r.get('mode',''):<18} | {raw:<8} | {sent:<8} | {saved:<8} | {pct:<6.1f}% | {lat:.2f}s")

by_task_mode = defaultdict(lambda: {"raw": 0, "sent": 0, "latencies": [], "turns": 0})
for r in records:
    key = (r["task_id"], r["mode"])
    by_task_mode[key]["raw"] += r.get("raw_prompt_tokens", 0)
    by_task_mode[key]["sent"] += r.get("sent_prompt_tokens", 0)
    by_task_mode[key]["latencies"].append(r.get("latency_seconds", 0))
    by_task_mode[key]["turns"] += 1

print("\n" + "="*95)
print(f"{'Task ID':<28} | {'Condition':<22} | {'Turns':<5} | {'Raw Tokens':<11} | {'Sent Tokens':<11} | {'Tokens Saved':<18} | {'Avg Latency'}")
print("="*95)

total_raw_ctrl = 0
total_sent_ctrl = 0
total_raw_treat = 0
total_sent_treat = 0

for (task, mode), data in sorted(by_task_mode.items()):
    saved = max(0, data["raw"] - data["sent"])
    pct = (saved / max(1, data["raw"])) * 100
    avg_lat = sum(data["latencies"]) / max(1, len(data["latencies"]))
    mode_label = "Treatment (System-1)" if mode == "with_system_1" else "Control (Full Context)"
    print(f"{task:<28} | {mode_label:<22} | {data['turns']:<5d} | {data['raw']:<11,d} | {data['sent']:<11,d} | {saved:<8,d} ({pct:5.2f}%) | {avg_lat:.2f}s")
    
    if mode == "with_system_1":
        total_raw_treat += data["raw"]
        total_sent_treat += data["sent"]
    else:
        total_raw_ctrl += data["raw"]
        total_sent_ctrl += data["sent"]

print("="*95)
total_saved_treat = max(0, total_raw_treat - total_sent_treat)
overall_pct = (total_saved_treat / max(1, total_raw_treat)) * 100
print(f"Total Treatment Input Volume Processed: {total_raw_treat:,} tokens")
print(f"Total Treatment Input Volume Transmitted: {total_sent_treat:,} tokens")
print(f"Total Tokens Saved by System-1 Memory: {total_saved_treat:,} tokens ({overall_pct:.2f}% aggregate reduction)")
