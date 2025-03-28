import numpy as np
from pyACA import ToolFreq2Midi

#TODO function and variable naming as mentioned in the other comments
def eval_pitchtrack(estimate_in_hz, groundtruth_in_hz, mode='pitch'):
   
    estimate = np.asarray(estimate_in_hz)
    gt = np.asarray(groundtruth_in_hz)
    
    # Create a mask for valid frames where ground truth frequency is not zero.
    valid_mask = gt != 0
    if np.sum(valid_mask) == 0:
        raise ValueError("No valid frames with nonzero ground truth frequency.")
    
    # Select only the valid entries
    estimate_valid = estimate[valid_mask]
    gt_valid = gt[valid_mask]
    
    # Compute the error in cents for each valid frame.
    # The formula for cents is: 1200 * log2(estimate / groundtruth)
    if mode == 'pitch':
        estimate_valid = ToolFreq2Midi(estimate_valid)
        gt_valid = ToolFreq2Midi(gt_valid)
    error_cents = 1200 * np.log2(estimate_valid / gt_valid)
    
    # Compute the RMS error (root mean square error)
    rms_error = np.sqrt(np.mean(np.square(error_cents)))
    
    return rms_error
    
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