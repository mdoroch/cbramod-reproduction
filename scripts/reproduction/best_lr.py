"""Print the EEG-SimpleConv lr with the best val AUROC from e2_simpleconv_lr.sh."""
import glob
import json

runs = [json.load(open(path)) for path in glob.glob("results/e2_lr_*/seed0.json")]
if not runs:
    raise SystemExit("no lr search results: run scripts/reproduction/e2_simpleconv_lr.sh first")
print(max(runs, key=lambda r: r["val"]["auroc"])["config"]["lr"])
