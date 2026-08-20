import json
import time
import urllib.request
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with CONFIG_PATH.open(encoding="utf-8") as config_file:
    OLLAMA_URL = json.load(config_file)["model_server_url"]

# The exact tags to pull in Ollama before running:
MODELS_TO_TEST = ["qwen3.5:35b", "qwen3.8:27b","qwen3.5:9b","qwen3-coder:30b", "qwen2.5-coder:14b","gemma4:12b","gemma4:31b"]

TEST_PROMPT = """
You are a financial data agent. Take the following raw budget log and write a Python script 
to extract monthly recurring software subscriptions, calculate the net burn rate, and output 
a structured JSON summary. Think through your reasoning steps clearly before writing the code.

Data Log:
2026-01-01: Deposit Payroll $5000
2026-01-02: Rent -$1500
2026-01-03: OpenAI API -$45
2026-01-04: GitHub Copilot -$10
2026-01-15: Grocery Target -$230
2026-02-01: Deposit Payroll $5000
2026-02-02: Rent -$1500
2026-02-05: OpenAI API -$62
2026-02-10: AWS Cloud -$120
"""

def run_benchmark(model_name):
    print("\n" + "="*60)
    print(f"[+] BENCHMARKING MODEL: {model_name}")
    print("="*60)
    print("[-] Processing prompt... Please wait...")
    
    payload = {
        "model": model_name,
        "prompt": TEST_PROMPT,
        "stream": False,
        "options": {
            "num_predict": 250  # Increased token limit so you can see full code outputs
        }
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
    
    start_time = time.time()
    try:
        with urllib.request.urlopen(req) as response:
            res_body = json.loads(response.read().decode("utf-8"))
            end_clock = time.time()
            
            # Extract response text and statistics
            response_text = res_body.get("response", "")
            eval_count = res_body.get("eval_count", 0)
            eval_duration = res_body.get("eval_duration", 1) / 1e9 
            
            tokens_per_second = eval_count / eval_duration if eval_duration > 0 else 0
            total_clock = end_clock - start_time
            
            # Print the actual text generation for auditing
            print("\n** GENERATED OUTPUT:")
            print("-" * 40)
            print(response_text)
            print("-" * 40)
            
            print(f"✅ Performance Metrics for {model_name}:")
            print(f"    - Speed: {tokens_per_second:.2f} tokens/sec")
            print(f"    - Generation Time: {eval_duration:.2f} seconds")
            print(f"    - Total Generated Tokens: {eval_count}")
            print(f"    - Total Latency: {total_clock:.2f} seconds")
            return tokens_per_second
    except Exception as e:
        print(f"❌ Skipped {model_name}: Check if pulled or if Ollama is running.")
        return None

if __name__ == "__main__":
    results = {}
    for model in MODELS_TO_TEST:
        speed = run_benchmark(model)
        if speed: results[model] = speed
            
    print("\n" + "="*70)
    print("         FINAL SPEED SUMMARY            ")
    print("="*70)
    for mod, speed in results.items():
        # Dynamic Evaluation Metric Mapping Logic
        if speed >= 45:
            tier = "🟢 EXCELLENT (Fits fully in VRAM, peak fluidity)"
        elif speed >= 30:
            tier = "🔵 GOOD      (Fits in VRAM, highly usable for coding)"
        elif speed >= 15:
            tier = "🟡 FAIR      (MoE layout balancing partial overflow)"
        elif speed >= 5:
            tier = "🟠 BAD       (Throttled heavily by DDR4 system RAM)"
        else:
            tier = "🔴 IMPOSSIBLE (System choked; locks up development flow)"
            
        print(f" * {mod:<20} : {speed:>6.2f} tokens/sec  ->  {tier}")
        
    print("="*70)