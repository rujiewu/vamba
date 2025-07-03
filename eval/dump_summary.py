import json
from eval_videomme import compute_summary

results_path = "/home/rujiewu/code/bigai_ml/rujie/code/vamba/res/Vamba-Qwen2-VL-7B/512frames/Vamba-Qwen2-VL-7B.json"
summary_path = "/home/rujiewu/code/bigai_ml/rujie/code/vamba/res/Vamba-Qwen2-VL-7B/512frames/Vamba-Qwen2-VL-7B_summary.json"

with open(results_path, "r") as f:
    merged = json.load(f)

summary = compute_summary(merged)

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

print(f"Summary saved to {summary_path}")