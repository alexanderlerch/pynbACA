import numpy as np
from pyACA import ToolFreq2Midi

#TODO function and variable naming as mentioned in the other comments
def eval_pitchtrack(estimate, groundtruth, mode='pitch'):
   
    estimate = np.asarray(estimate)
    gt = np.asarray(groundtruth)
  
    if mode == 'pitch':
        error = estimate - gt

    if mode == 'freq':
        # Direct difference in Hz
        error = estimate - gt
        
    # Compute the RMS error (root mean square error)
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