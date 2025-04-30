import json


# predcitions are under results["samples"][TASK_NAME]
def dump_predictions(results, output_dir):
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(results["samples"], f, indent=2, default=str)
