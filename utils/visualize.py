import matplotlib.pyplot as plt
import numpy as np
import pyACA


def visualAudioBlock(blockTime, origAudio, window_size, fs):
    duration = 0.5
    offset = 1.0
    window_size_in_second = window_size / fs
    n_frame = int(fs * duration)
    t = np.linspace(offset, offset + duration, n_frame, endpoint=False)
    
    # Extract the audio segment for plotting
    audio_segment = origAudio[int(offset * fs):int(offset * fs + n_frame)]

    # Plot duplicated waveforms
    fig, axes = plt.subplots(2, 1, figsize=(24, 8), sharex=True)
    toggle = False
    for idx, ax in enumerate(axes):
        ax.plot(t, audio_segment, label="Original Waveform", alpha=0.7)
        
        # Overlay block segments
        toggle = not toggle
        added_labels = set()  # Track which labels have been added
        
        for i, middle_time in enumerate(blockTime):
            start, end = middle_time - window_size_in_second / 2, middle_time + window_size_in_second / 2
            if end < offset:
                continue
            if start > offset + duration:
                break
            
            # Differentiate odd/even windows
            if (idx == 0 and i % 2 == 1) or (idx == 1 and i % 2 == 0):
                continue
            
            color = "y" if toggle else "r"
            label = "Odd Windows" if (idx == 0 and color not in added_labels) else ("Even Windows" if (idx == 1 and color not in added_labels) else "")
            
            ax.axvspan(start, end, color=color, alpha=0.3, label=label)
            added_labels.add(color)
        
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{'Odd' if idx == 0 else 'Even'} Numbered Windows Overlap")
        ax.legend()

    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def visualizeSTFT(stft, sr, hop_length=512):
    # Compute the magnitude in dB (add a small constant to avoid log(0))
    magnitude_db = 20 * np.log10(np.abs(stft) + 1e-6)
    
    n_freq_bins, n_time_frames = stft.shape
    time_axis = np.arange(n_time_frames) * hop_length / sr
    freq_axis = np.linspace(0, sr/2, n_freq_bins)
    
    # Define the extent: [xmin, xmax, ymin, ymax]
    extent = [time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
    
    plt.figure(figsize=(10, 6))
    plt.imshow(magnitude_db, aspect='auto', origin='lower', cmap='viridis', extent=extent)
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('STFT Magnitude (dB)')
    plt.show()


def visualizePitchTracking(gt_time, gt_freq, est_time=None, est_freq=None):

    plt.figure(figsize=(10, 6))
    plt.plot(gt_time, gt_freq, linestyle='-', color='b', label='Ground Truth')
    if est_freq is not None and est_time is not None:
        plt.plot(est_time, est_freq, linestyle='--', color='r', label='Estimated')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Pitch Tracking Visualization')
    plt.grid(True)
    plt.legend()
    plt.show()


import time
from utils.eval import eval_pitchtrack

def visualizeMultiPitchAlgo(wav, sr, gt_qfreq, 
                               algorithms=None,
                               iBlockLength=1024, iHopLength=512):
 
    
    if algorithms is None:
        algorithms = ['SpectralAcf', 'SpectralHps', 'TimeAcf', 
                      'TimeAmdf', 'TimeAuditory', 'TimeZeroCrossings']
    
    n_algos = len(algorithms)
    nrows = 2
    ncols = 3
    plt.figure(figsize=(18, 10))
    
    # Ensure ground truth is a numpy array
    gt_qfreq = np.asarray(gt_qfreq)
    
    for idx, algo in enumerate(algorithms):
        # Compute pitch and measure runtime
        start_time = time.time()
        est_freq, est_time = pyACA.computePitch(algo, wav, sr, 
                                                iBlockLength=iBlockLength, 
                                                iHopLength=iHopLength)
        runtime = time.time() - start_time
        
        # Compute RMS error in cents between estimated and ground truth pitch
        rms_error = eval_pitchtrack(estimate_in_hz=est_freq, 
                                    groundtruth_in_hz=gt_qfreq, 
                                    mode='pitch')
        
        # Create copies for plotting that replace points with np.nan where either is zero.
        # This prevents connecting segments across gaps.
        plot_gt = np.copy(gt_qfreq)
        plot_est = np.copy(est_freq)
        mask = (gt_qfreq == 0) | (est_freq == 0)
        if mask.any():
            plot_gt[mask] = np.nan
            plot_est[mask] = np.nan
        
        # Plot the pitch contours
        ax = plt.subplot(nrows, ncols, idx + 1)
        ax.plot(est_time, plot_est, label='Estimated', color='b')
        ax.plot(est_time, plot_gt, label='Ground Truth', color='r', linestyle='--')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f"{algo}\nRMS Error: {rms_error:.2f} cents, Run time: {runtime:.3f} s")
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()