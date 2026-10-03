"""Run the published common-start cohesion-identification comparison."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RUNS = ("synthetic_state_heads", "synthetic_state_heads_continued", "synthetic_state_heads_virtual")


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / "code" / script), *map(str, args)],
                   cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check", "common-start", "control", "virtual-work", "evaluate", "all"])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.stage == "check":
        sys.path.insert(0, str(ROOT / "code"))
        from run_mixed_pinn import specimen, make_core
        data = specimen("m-1-1", 12)
        core = make_core(data, 10., 3.)
        s, n = data["local_s_mm"], data["local_n_mm"]
        assert data["points_mm"].shape == (401, 3)
        assert ((abs(s) <= 4) & (abs(n) >= 4)).sum() == 97
        assert ((abs(s) >= 8) & (abs(n) >= 4)).sum() == 93
        assert core.model.mesh_info["nodes"] == 1083
        assert core.model.mesh_info["band_tetrahedra"] == 1139
        print("Published inputs and mesh checked:", core.model.mesh_info)
        return
    shared = ["--mode", "synthetic", "--frames", "12", "--mesh-size", "10",
              "--band-size", "3", "--equilibrated-fields", "--state-heads",
              "--physics-weight", "100", "--observation-scale-mm", "0.002",
              "--physics-normalization", "final_load", "--device", args.device]
    if args.stage in {"common-start", "all"}:
        run("run_mixed_pinn.py", *shared, "--steps", "700", "--lbfgs-steps", "250",
            "--output", "results/mixed_pinn/" + RUNS[0])
    for stage, name, weight in [("control", RUNS[1], "0"), ("virtual-work", RUNS[2], "100")]:
        if args.stage in {stage, "all"}:
            run("run_mixed_pinn.py", *shared, "--steps", "0", "--lbfgs-steps", "350",
                "--observation-virtual-weight", weight,
                "--warm-start", "results/mixed_pinn/" + RUNS[0] + "/checkpoint.pt",
                "--output", "results/mixed_pinn/" + name)
    if args.stage in {"evaluate", "all"}:
        for name in RUNS:
            run("diagnose_synthetic_material.py", "--result", "results/mixed_pinn/" + name)
        run("prepare_paper_inputs.py")
        run("analyze_paper_evidence.py")


if __name__ == "__main__":
    main()
