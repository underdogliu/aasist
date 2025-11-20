import sys
import os
import json
import argparse
import numpy as np
import csv

import soundfile as sf
import torch

from main import get_model
from data_utils import pad, pad_random


def generate_scoring_file(input_csv, config, output_file, device="cuda"):
    """
    Reads the CSV, runs inference, and writes the score file.
    Output format: ID score label
    """
    with open(config, "r") as f_json:
        config = json.loads(f_json.read())
    
    model_config = config["model_config"]
    model = get_model(model_config, device)
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    model.eval()

    print(f"Processing {input_csv}...")    
    results = []
    #with torch.no_grad():
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            utt_id = row['ID']
            label = row['Label']
            file_path = row['Path']
            X, _ = sf.read(file_path)
            X_pad = pad(X, 64600)
            x_inp = torch.Tensor(X_pad).unsqueeze(0).to(device)
            with torch.no_grad():
                _, output = model(x_inp)
                score = output.squeeze()[1].detach().cpu().numpy()
            results.append(f"{utt_id} {score} {label}")
    # Write to output text file
    with open(output_file, 'w') as f:
        for line in results:
            f.write(line + "\n")            
    print(f"Scoring file saved to: {output_file}")


def compute_metrics(score_file, acc_threshold=0.0):
    """
    Loads the generated score file and computes EER and Accuracy.
    Assumes format: ID score label
    """
    # Load scores
    data = np.genfromtxt(score_file, dtype=str)
    
    # Handle case where file might be empty or single line
    if data.ndim == 1 and data.size > 0:
        data = data.reshape(1, -1)
    elif data.size == 0:
        print("Error: Score file is empty.")
        return

    # data columns: 0=ID, 1=Score, 2=Label
    scores = data[:, 1].astype(float)
    labels = data[:, 2]

    # Separate scores by label
    # Assuming 'real' is the target (bonafide) class
    bonafide_scores = scores[labels == 'real']
    spoof_scores = scores[labels != 'real']

    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        print("Error: Need both 'real' and 'fake' samples to compute EER.")
        return

    # Compute EER
    eer, threshold = compute_eer(bonafide_scores, spoof_scores)
    
    # Compute Accuracy at EER Threshold
    # Logic: Score > Threshold => Predicted Real
    # Logic: Score < Threshold => Predicted Fake
    
    correct_real = np.sum(bonafide_scores >= acc_threshold)
    correct_spoof = np.sum(spoof_scores < acc_threshold)
    total_samples = len(bonafide_scores) + len(spoof_scores)
    
    accuracy = (correct_real + correct_spoof) / total_samples

    # Report ONLY EER and Accuracy
    print("-" * 30)
    print(f"CM SYSTEM EVALUATION")
    print("-" * 30)
    print(f"EER      : {eer * 100:.3f} %")
    print(f"Threshold: {acc_threshold}")
    print(f"Accuracy : {accuracy * 100:.3f} %")
    print("-" * 30)


def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - \
        (np.arange(1, n_scores + 1) - tar_trial_sums)

    # false rejection rates
    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums /
                          nontarget_scores.size))  # false acceptance rates
    # Thresholds are the sorted scores
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Audio Deepfake Detection Model")
    
    parser.add_argument("--input_csv", type=str, required=True, 
                        help="Path to input CSV file (must contain 'ID', 'Path', 'Label')")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_score", type=str, required=True)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generate_scoring_file(args.input_csv, args.config, args.output_score, device=device)
    
    compute_metrics(args.output_score)
