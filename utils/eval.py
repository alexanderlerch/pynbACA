import numpy as np
from pyACA import ToolFreq2Midi

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
    
