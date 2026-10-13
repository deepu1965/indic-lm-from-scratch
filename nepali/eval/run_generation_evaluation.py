import os
import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run only on a tiny subset (5 docs) for validation")
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = "nepali/eval/results"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "generation_metrics.csv")
    
    if os.path.exists(out_csv):
        os.remove(out_csv)

    num_examples = 5 if args.smoke_test else args.num_examples
    
    # 0.0 is used to represent greedy decoding
    temperatures = [0.0, 0.5, 1.0, 1.5]

    for temp in temperatures:
        temp_str = "greedy" if temp == 0.0 else f"temp_{temp}"
        out_json = os.path.join(out_dir, f"generation_metrics_{temp_str}.json")
        
        cmd = [
            sys.executable, "-m", "nepali.eval.generate_eval",
            "--model-cfg", "nepali/configs/model_config.yaml",
            "--checkpoint", "nepali/outputs/checkpoints/best_model.pt",
            "--tokenized-data", "nepali/corpus/tokenised_splits/test.jsonl",
            "--tokenizer-json", "nepali/tokenizer/models/bpe_v10000.json",
            "--out-json", out_json,
            "--out-csv", out_csv,
            "--num-examples", str(num_examples),
            "--temperature", str(temp)
        ]
        
        print(f"\n============================================================")
        print(f"Running generation evaluation: NEPALI | Config: {temp_str}")
        print(f"============================================================")
        
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Error during generation evaluation of nepali with temp={temp}")
            sys.exit(result.returncode)

    print("\nAll generation evaluations completed successfully!")
    print(f"Results saved to {out_dir}/")

if __name__ == "__main__":
    main()
