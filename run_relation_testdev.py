
import argparse
import ast
import json
import shutil
from pathlib import Path
import mobilevlm_relation_kd as kd

saved = Path("/kaggle/working/muzna_relation_kd_main")
out = Path("/kaggle/working/muzna_relation_kd_testdev")
out.mkdir(exist_ok=True)

shutil.copytree(
    saved / "relation_adapter_epoch_1",
    out / "relation_adapter_epoch_1",
    dirs_exist_ok=True
)
shutil.copy2(
    saved / "training_complete.json",
    out / "training_complete.json"
)

matches = list(Path("/kaggle/input").rglob("gqa_v2_seed42"))
assert len(matches) == 1, f"Dataset folders found: {matches}"

args = argparse.Namespace(
    shared_dir=str(saved),
    data_dir=str(matches[0]),
    output_dir=str(out),
    profile="main",
    lambda_relation=1.0,
    repo_dir="/kaggle/working/MobileVLM",
    gt_baseline_csv=None
)

cfg, lock, _, _ = kd.setup(args)
test = json.loads((matches[0] / "testdev.json").read_text())
assert len(test) == 12578, f"Expected 12578 questions, got {len(test)}"
assert len({row["id"] for row in test}) == len(test)
assert all((matches[0] / row["image"]).is_file() for row in test)

cfg["validation_file"] = "testdev"
(out / "testdev_question_ids.txt").write_text(
    "\n".join(row["id"] for row in test)
)

source = Path(kd.__file__).read_text()
tree = ast.parse(source)
node = next(
    n for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "evaluate"
)
function = ast.get_source_segment(source, node)
function = function.split("    comparisons = []")[0]
function = function.replace(
    "relation_kd_validation", "relation_kd_testdev"
).replace("Relation KD validation", "Relation KD testdev")
function += "\n    print(json.dumps(metrics, indent=2), flush=True)\n"
exec(compile(function, "<testdev evaluator>", "exec"), kd.__dict__)

print("Evaluating saved adapter on 12,578 testdev questions.", flush=True)
kd.evaluate(cfg, lock, test, args)
