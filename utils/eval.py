import numpy as np
from pyACA import ToolFreq2Midi

#TODO function and variable naming as mentioned in the other comments
def eval_pitchtrack(estimate, groundtruth, mode='pitch'):
   
    estimate = np.asarray(estimate)
    gt = np.asarray(groundtruth)
  
    if mode == 'pitch':
        error = estimate - gt
        rms_error = np.sqrt(np.mean(np.square(error))) * 100

    if mode == 'freq':
        # Direct difference in Hz
        error = estimate - gt
        rms_error = np.sqrt(np.mean(np.square(error)))
            
    return rms_error

# return cents error
def eval_pitchtrack_midi(estimate, groundtruth):
  est = np.asarray(estimate, dtype=float).squeeze()
  gt  = np.asarray(groundtruth, dtype=float).squeeze()
  valid = np.isfinite(est) & np.isfinite(gt)
  valid &= (gt != 0)
  error_cent = 100 * (est[valid] - gt[valid])
  rms_error_cent = np.sqrt(np.mean(np.square(error_cent)))
  return rms_error_cent

def eval_midi_est_accuracy(estimate, groundtruth):
    est = np.asarray(estimate, dtype=float).squeeze()
    gt  = np.asarray(groundtruth, dtype=float).squeeze()

    # Valid frames: finite and non-zero ground truth
    valid = np.isfinite(est) & np.isfinite(gt)
    valid &= (gt != 0)

    if not np.any(valid):
        return np.nan, np.nan

    # Round estimate to nearest integer MIDI
    est_rounded = np.rint(est).astype(float)

    # Differences in semitones on valid frames
    diff_semitones = est_rounded[valid] - gt[valid]

    # Accuracy with octave errors counted 
    # Correct if rounded estimate equals ground truth
    correct_all = (est_rounded[valid] == gt[valid])
    acc_with_octave = np.sum(correct_all) / correct_all.size

    # Accuracy without octave errors 
    # Octave error = non-zero multiple of 12 semitones
    is_octave_error = (np.mod(diff_semitones, 12) == 0) & (diff_semitones != 0)

    non_octave_mask = ~is_octave_error
    if not np.any(non_octave_mask):
        acc_without_octave = np.nan
    else:
        correct_non_oct = correct_all & non_octave_mask
        acc_without_octave = np.sum(correct_non_oct) / np.sum(non_octave_mask)

    return acc_with_octave, acc_without_octave

def computeTemporalFmeasure(est_onsets, ref_onsets, tolerance=0.05):
    """Compute F-measure between estimated and reference onsets
    
    Args:
        est_onset: estimated onset times in seconds
        ref_onset: reference onset times in seconds 
        tolerance: tolerance window in seconds (default: 50ms)
    
    Returns:
        f_measure: F-measure score
        precision: precision score
        recall: recall score
    """
    # Initialize counters
    true_positives = 0
    
    # Count matches within tolerance window
    matched_ref = np.zeros(len(ref_onsets), dtype=bool)
    for ref in ref_onsets:
        # Find closest estimated onset to each reference
        distances = np.abs(est_onsets - ref)
        if np.min(distances) <= tolerance and not matched_ref[np.argmin(distances)]:
            true_positives += 1
            # delete the matched est onset
            est_onsets = np.delete(est_onsets, np.argmin(distances))
            matched_ref[np.argmin(distances)] = True
            
    # Calculate metrics
    if len(est_onsets) == 0:
        precision = 0
    else:
        precision = true_positives / len(est_onsets)
        
    if len(ref_onsets) == 0:
        recall = 0
    else:
        recall = true_positives / len(ref_onsets)
    
    # Calculate F-measure
    if precision + recall == 0:
        f_measure = 0
    else:
        f_measure = 2 * precision * recall / (precision + recall)
        
    return f_measure, precision, recall


# Ordered pitch classes for mapping semitones
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def key_relation_score(pred_key, true_key):
    tonic_pred, mode_pred = pred_key.split()
    tonic_true, mode_true = true_key.split()

    idx_pred = PITCH_CLASSES.index(tonic_pred)
    idx_true = PITCH_CLASSES.index(tonic_true)
    diff = (idx_pred - idx_true) % 12

    # Exact match
    if pred_key == true_key:
        return 1.0
    # Perfect fifth relation (±7 semitones)
    elif diff in [5, 7] and mode_pred == mode_true:
        return 0.5
    # Relative major/minor
    elif (mode_pred == "minor" and mode_true == "major" and diff == 9) or (mode_pred == "major" and mode_true == "minor" and diff == 3):
        return 0.3
    # Parallel major/minor
    elif tonic_pred == tonic_true and mode_pred != mode_true:
        return 0.2
    # Otherwise unrelated
    return 0.0

def evaluate_key_detection(pred_keys, true_keys):
    N = len(true_keys)

    tonic_correct = 0
    mode_correct = 0
    mirex_total = 0.0

    for pred, true in zip(pred_keys, true_keys):
        tonic_pred, mode_pred = pred.split()
        tonic_true, mode_true = true.split()
        # Tonic accuracy
        if tonic_pred == tonic_true:
            tonic_correct += 1
        # Mode accuracy
        if mode_pred == mode_true:
            mode_correct += 1
        # Weighted score (MIREX)
        mirex_total += key_relation_score(pred, true)

    tonic_acc = tonic_correct / N
    mode_acc = mode_correct / N
    mirex_score = mirex_total / N

    return {
        "tonic_acc": tonic_acc,
        "mode_acc": mode_acc,
        "mirex_score": mirex_score
    }
    
def normalize_key_label(key_str):
    if not key_str:
        return None
    parts = key_str.strip().split()
    if len(parts) < 2:
        return key_str
    tonic, mode = parts[0], parts[1]
    tonic = tonic[0].upper() + tonic[1:] if len(tonic) > 1 else tonic.upper()
    mode = "major" if mode.lower() in ["maj", "major"] else "minor"
    return f"{tonic} {mode}"