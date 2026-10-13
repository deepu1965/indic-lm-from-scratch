import os
import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run only on a tiny subset for validation")
    args = parser.parse_args()

    out_dir = "hindi/eval/results"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "intrinsic_comparison.csv")
    
    if os.path.exists(out_csv):
        os.remove(out_csv)

    tasks = [
        {
            "split": "val",
            "model_cfg": "hindi/configs/model_config.yaml",
            "checkpoint": "hindi/outputs/checkpoints/best_model.pt",
            "raw_data": "hindi/corpus/splits/val.jsonl",
            "tokenized_data": "hindi/corpus/tokenised_splits/val.jsonl",
            "tokenizer_json": "hindi/tokenizer/models/bpe_v10000.json",
            "out_json": os.path.join(out_dir, "validation_metrics.json")
        },
        {
            "split": "test",
            "model_cfg": "hindi/configs/model_config.yaml",
            "checkpoint": "hindi/outputs/checkpoints/best_model.pt",
            "raw_data": "hindi/corpus/splits/test.jsonl",
            "tokenized_data": "hindi/corpus/tokenised_splits/test.jsonl",
            "tokenizer_json": "hindi/tokenizer/models/bpe_v10000.json",
            "out_json": os.path.join(out_dir, "test_metrics.json")
        }
    ]

    for task in tasks:
        cmd = [
            sys.executable, "-m", "hindi.eval.intrinsic_eval",
            "--split", task['split'],
            "--model-cfg", task["model_cfg"],
            "--checkpoint", task["checkpoint"],
            "--raw-data", task["raw_data"],
            "--tokenized-data", task["tokenized_data"],
            "--tokenizer-json", task["tokenizer_json"],
            "--out-json", task["out_json"],
            "--out-csv", out_csv
        ]
        
        if args.smoke_test:
            cmd.append("--smoke-test")
            
        print(f"\n============================================================")
        print(f"Running evaluation: HINDI {task['split'].upper()}")
        print(f"============================================================")
        
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Error during evaluation of hindi {task['split']}")
            sys.exit(result.returncode)

    print("\nAll evaluations completed successfully!")
    print(f"Results saved to {out_dir}/")

if __name__ == "__main__":
    main()
