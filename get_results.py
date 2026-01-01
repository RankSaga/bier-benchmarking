"""
Helper script to retrieve results from Modal and save locally.
Run: python get_results.py
"""
import json
import subprocess
from pathlib import Path

def get_modal_results():
    """Get results from Modal function."""
    print("Retrieving results from Modal...")
    try:
        result = subprocess.run(
            ["modal", "run", "modal_app.py::get_results"],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse the JSON output (Modal returns JSON)
        output = result.stdout
        # Extract JSON from output (might have extra text)
        json_start = output.find('{')
        json_end = output.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = output[json_start:json_end]
            results = json.loads(json_str)
            return results
        else:
            print("Could not parse JSON from output")
            print("Output:", output)
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error running Modal command: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None


def save_results_locally(results, output_path):
    """Save results to local file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {output_path}")


if __name__ == "__main__":
    print("="*60)
    print("Retrieving Benchmark Results from Modal")
    print("="*60)
    
    results = get_modal_results()
    
    if results:
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            print("\n💡 Alternative: Access results via Modal dashboard:")
            print("   https://modal.com/apps -> ranksaga-beir-benchmark")
        else:
            output_path = Path("results/baseline/e5_base_v2_results.json")
            save_results_locally(results, output_path)
            print("\n📊 Results summary:")
            if "results" in results:
                for dataset, metrics in results["results"].items():
                    if "error" not in metrics:
                        print(f"  {dataset}: NDCG@10 = {metrics.get('NDCG@10', 'N/A'):.4f}")
    else:
        print("\n❌ Failed to retrieve results")
        print("\n💡 Try accessing via Modal dashboard:")
        print("   https://modal.com/apps -> ranksaga-beir-benchmark -> Volumes")

