from pathlib import Path

import numpy as np

from src.data import SEED, load_dry_bean_splits
from src.train import train, evaluate
from src.plotting import save_heatmap, plot_side_by_side

ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "results" / "logs"
HEATMAP_DIR = ROOT / "results" / "heatmaps"

ACTIVATIONS = ["sigmoid", "relu"]
CLIP_NORMS = [None, 1.0]
SEEDS = [SEED, 123]
NUM_STEPS = 2000
RECORD_EVERY = 10


# short label of a clip setting, e.g. "noclip" or "clip1.0"
def clip_label(clip_norm):
    if clip_norm is None:
        return "noclip"
    return f"clip{clip_norm}"


# readable run name used for file names, e.g. "relu_clip1.0_seed42" or "sigmoid_noclip_seed123"
def make_run_name(activation, clip_norm, seed):
    # build the string from the three arguments
    return f"{activation}_{clip_label(clip_norm)}_seed{seed}"


# one training run, returns the result dict of train() plus train and test accuracy
def run_single_experiment(activation, clip_norm, seed, train_set, test_set):
    # train on the train set only
    train_result = train(activation, clip_norm, seed, num_steps=NUM_STEPS, record_every=RECORD_EVERY, train_set=train_set)
    # evaluate on the trained model (eval mode + no_grad happen inside evaluate)
    # the gap between train and test accuracy shows overfitting
    train_result["train_accuracy"] = evaluate(train_result["model"], train_set)
    train_result["test_accuracy"] = evaluate(train_result["model"], test_set)
    return train_result


# save the numbers of one run so plots can be redone without retraining
def save_result(result, run_name):
    # create LOGS_DIR if it does not exist
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    # one file each for grad_matrix, pre_clip_norms and losses
    np.save(LOGS_DIR / f"{run_name}_grad_matrix.npy", result["grad_matrix"])
    np.save(LOGS_DIR / f"{run_name}_pre_clip_norms.npy", result["pre_clip_norms"])
    np.save(LOGS_DIR / f"{run_name}_losses.npy", result["losses"])


# run the full grid of 2 activations x 2 clip settings x 2 seeds = 8 runs
# returns a dict keyed by make_run_name(activation, clip_norm, seed)
def run_all_experiments():
    # final result dict keyed by make_run_name
    results = {}
    # load the data once and reuse it for every run
    train_set, test_set = load_dry_bean_splits()
    # loop over ACTIVATIONS, CLIP_NORMS and SEEDS
    for activation in ACTIVATIONS:
        for clip_norm in CLIP_NORMS:
            for seed in SEEDS:
                # for each combination call run_single_experiment(...) and save_result(...)
                result = run_single_experiment(activation, clip_norm, seed, train_set, test_set)
                run_name = make_run_name(activation, clip_norm, seed)
                save_result(result, run_name)
                results[run_name] = result
    return results


# task step 4: sigmoid vs relu next to each other (same seed, same clip setting)
def plot_activation_comparison(results, seed, clip_norm):
    # pick the grad_matrix of every activation out of results
    matrices = [results[make_run_name(activation, clip_norm, seed)]["grad_matrix"] for activation in ACTIVATIONS]
    # panel titles, a title and a one-sentence caption
    panel_titles = ["Sigmoid hidden layers", "ReLU hidden layers"]
    title = f"Gradient flow: sigmoid vs ReLU ({clip_label(clip_norm)}, seed {seed})"
    caption = "Per-layer gradient norm during training for sigmoid and ReLU hidden layers, both panels share one color scale."
    # save into HEATMAP_DIR with a readable file name
    path = HEATMAP_DIR / f"activation_comparison_{clip_label(clip_norm)}_seed{seed}.png"
    plot_side_by_side(matrices, panel_titles, title, caption, path, RECORD_EVERY)


# task step 5: without vs with clipping (same activation, same seed)
def plot_clipping_comparison(results, activation, seed):
    # pick the unclipped and the clipped grad_matrix out of results
    matrices = [results[make_run_name(activation, clip_norm, seed)]["grad_matrix"] for clip_norm in CLIP_NORMS]
    panel_titles = ["No clipping" if clip_norm is None else f"Clipped at max norm {clip_norm}" for clip_norm in CLIP_NORMS]
    title = f"Gradient flow: effect of clipping ({activation}, seed {seed})"
    caption = "Per-layer gradient norm without and with global-norm clipping, norms are measured after clipping and both panels share one color scale."
    path = HEATMAP_DIR / f"clipping_comparison_{activation}_seed{seed}.png"
    plot_side_by_side(matrices, panel_titles, title, caption, path, RECORD_EVERY)


# task step 6: the same setting under both seeds next to each other
def plot_seed_comparison(results, activation, clip_norm):
    # pick the grad_matrix of every seed out of results
    matrices = [results[make_run_name(activation, clip_norm, seed)]["grad_matrix"] for seed in SEEDS]
    panel_titles = [f"Seed {seed}" for seed in SEEDS]
    title = f"Gradient flow across seeds ({activation}, {clip_label(clip_norm)})"
    caption = "The same experiment under two random seeds, a pattern that appears in both panels is not a one-off."
    path = HEATMAP_DIR / f"seed_comparison_{activation}_{clip_label(clip_norm)}.png"
    plot_side_by_side(matrices, panel_titles, title, caption, path, RECORD_EVERY)


# print the numbers needed for the README: accuracy, mean norm per layer, does the clip threshold trigger
def print_summary(results):
    for activation in ACTIVATIONS:
        for clip_norm in CLIP_NORMS:
            for seed in SEEDS:
                result = results[make_run_name(activation, clip_norm, seed)]
                print(f"\n{make_run_name(activation, clip_norm, seed)}")
                # test accuracy with the train accuracy next to it
                print(f"  accuracy: test {result['test_accuracy']:.3f} | train {result['train_accuracy']:.3f}")
                # mean gradient norm per layer, layer 1 (input side) to layer 8 (output)
                print(f"  mean grad norm per layer: {np.array2string(result['grad_matrix'].mean(axis=0), precision=5)}")
                # share of recorded steps where the global norm exceeded the clip threshold
                if clip_norm is not None:
                    share = (result["pre_clip_norms"] > clip_norm).mean()
                    print(f"  clip threshold {clip_norm} exceeded in {share:.1%} of the recorded steps")


# entry point used by run_experiments.py
def main():
    results = run_all_experiments()
    # create HEATMAP_DIR if it does not exist
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    # call the three plot functions for every setting that is needed
    for seed in SEEDS:
        for clip_norm in CLIP_NORMS:
            plot_activation_comparison(results, seed, clip_norm)
    for activation in ACTIVATIONS:
        for seed in SEEDS:
            plot_clipping_comparison(results, activation, seed)
        for clip_norm in CLIP_NORMS:
            plot_seed_comparison(results, activation, clip_norm)
    print_summary(results)
