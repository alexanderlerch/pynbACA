import matplotlib.pyplot as plt
import numpy as np
import pyACA
import matplotlib.animation as animation

# def visualAudioBlock(blockTime, origAudio, window_size, fs):
#     duration = 0.5
#     offset = 1.0
#     window_size_in_second = window_size / fs
#     n_frame = int(fs * duration)
#     t = np.linspace(offset, offset + duration, n_frame, endpoint=False)
    
#     # Extract the audio segment for plotting
#     audio_segment = origAudio[int(offset * fs):int(offset * fs + n_frame)]

#     # Plot duplicated waveforms
#     fig, axes = plt.subplots(2, 1, figsize=(24, 8), sharex=True)
#     toggle = False
#     for idx, ax in enumerate(axes):
#         ax.plot(t, audio_segment, label="Original Waveform", alpha=0.7)
        
#         # Overlay block segments
#         toggle = not toggle
#         added_labels = set()  # Track which labels have been added
        
#         for i, middle_time in enumerate(blockTime):
#             start, end = middle_time - window_size_in_second / 2, middle_time + window_size_in_second / 2
#             if end < offset:
#                 continue
#             if start > offset + duration:
#                 break
            
#             # Differentiate odd/even windows
#             if (idx == 0 and i % 2 == 1) or (idx == 1 and i % 2 == 0):
#                 continue
            
#             color = "y" if toggle else "r"
#             label = "Odd Windows" if (idx == 0 and color not in added_labels) else ("Even Windows" if (idx == 1 and color not in added_labels) else "")
            
#             ax.axvspan(start, end, color=color, alpha=0.3, label=label)
#             added_labels.add(color)
        
#         ax.set_ylabel("Amplitude")
#         ax.set_title(f"{'Odd' if idx == 0 else 'Even'} Numbered Windows Overlap")
#         ax.legend()

#     plt.xlabel("Time (s)")
#     plt.tight_layout()
#     plt.show()

def visualAudioBlock(blockTime, origAudio, window_size, fs):
    
    # Settings for the visualization segment
    duration = 0.1  # duration (in seconds) of the segment to visualize
    offset = 1.0    # start time (in seconds) of the segment
    n_frame = int(fs * duration)
    n_frame = int(fs * duration)
    t = np.linspace(offset, offset + duration, n_frame, endpoint=False)
    
    # Extract a segment of the original audio for plotting
    audio_segment = origAudio[int(offset * fs): int(offset * fs + n_frame)]
    
    # Create a Hann window (normalized shape)
    hann_window = np.hanning(window_size)
    window_duration = window_size / fs  # duration of the window in seconds

    # Filter blockTime to only include blocks within the visualization segment
    valid_blockTimes = [bt for bt in blockTime if (bt >= offset) and (bt <= offset + duration)]
    
    # Select up to three windows
    selected_blockTimes = valid_blockTimes[:3]
    
    # Create 3 subplots (one per window)
    fig, axes = plt.subplots(3, 1, figsize=(24, 12), sharex=True)
    
    for idx, ax in enumerate(axes):
        # Plot the original audio segment
        ax.plot(t, audio_segment, label="Original Waveform", color="blue", alpha=0.7)
        
        if idx < len(selected_blockTimes):
            mid_time = selected_blockTimes[idx]
            start_time = mid_time - window_duration / 2
            end_time = mid_time + window_duration / 2
            
            # Create a time axis for the window and scale the Hann window for visualization
            time_window = np.linspace(start_time, end_time, window_size)
            # Scale the Hann window to the maximum absolute amplitude of the audio segment
            scaled_hann = hann_window * np.max(np.abs(audio_segment))
            
            # Overlay the Hann window shape on the waveform
            ax.plot(time_window, scaled_hann, color="red", linewidth=2, label="Hann Window")
            ax.axvspan(start_time, end_time, color="red", alpha=0.3)
            ax.set_title(f"Window {idx+1}: Centered at {mid_time:.2f} s")
        else:
            ax.set_title(f"Window {idx+1}: No Data")
        
        ax.set_ylabel("Amplitude")
        ax.legend()
    
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

def visualizeSpec(stft, sr, hop_length=512, log_magnitude=True, ax=None):
    # Compute the magnitude in dB (add a small constant to avoid log(0))
    magnitude = np.abs(stft)
    if log_magnitude:
        magnitude_to_plot = 20 * np.log10(magnitude + 1e-6)
        colorbar_label = 'Magnitude (dB)'
        title = 'STFT Magnitude (dB)'
    else:
        magnitude_to_plot = magnitude
        colorbar_label = 'Magnitude'
        title = 'STFT Magnitude'
        
    n_freq_bins, n_time_frames = stft.shape
    time_axis = np.arange(n_time_frames) * hop_length / sr
    freq_axis = np.linspace(0, sr/2, n_freq_bins)
    extent = [time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
    img = ax.imshow(magnitude_to_plot, aspect='auto', origin='lower', cmap='viridis', extent=extent)
    plt.colorbar(img, ax=ax, label=colorbar_label)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    return ax


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


def visualizeMelSpectrogram(M, f_c, t, title='Mel Spectrogram', colormap='viridis', 
                           figsize=(10, 6), min_dB=-60, vmin=None, vmax=None, 
                           show_colorbar=True, min_freq=20):
    
  
    
    fig, ax = plt.subplots(figsize=figsize)
    valid_freq_indices = np.where(f_c >= min_freq)[0]
    if len(valid_freq_indices) == 0:
        print(f"Warning: No frequencies above {min_freq}Hz found")
        valid_freq_indices = np.arange(len(f_c))
    
    f_c_filtered = f_c[valid_freq_indices]
    M_filtered = M[valid_freq_indices, :]
    
    if vmin is None:
        if np.min(M_filtered) < 0:  # Data seems to be in dB
            vmin = np.max(M_filtered) + min_dB
        else:
            vmin = np.min(M_filtered)
    
    if vmax is None:
        vmax = np.max(M_filtered)

    X, Y = np.meshgrid(t, f_c_filtered)
    im = ax.pcolormesh(X, Y, M_filtered, 
                      cmap=colormap,
                      vmin=vmin, vmax=vmax,
                      shading='auto')
    
    ax.set_yscale('log')
    ax.set_ylim(f_c_filtered[0], f_c_filtered[-1])
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    
    # Add colorbar
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax)
        if np.min(M_filtered) < 0:  # Data seems to be in dB
            cbar.set_label('Magnitude (dB)')
        else:
            cbar.set_label('Magnitude')
    if f_c_filtered[-1] > 1000:
        yticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        yticks = [f for f in yticks if f >= f_c_filtered[0] and f <= f_c_filtered[-1]]
        ax.set_yticks(yticks)

        ytick_labels = []
        for f in yticks:
            if f >= 1000:
                ytick_labels.append(f'{f/1000:.0f} kHz')
            else:
                ytick_labels.append(f'{f:.0f} Hz')
        ax.set_yticklabels(ytick_labels)
    else:
        # For smaller frequency ranges, use more appropriate tick values
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%d Hz'))
    ax.grid(which='minor', axis='y', linestyle=':', alpha=0.3)
    ax.grid(which='major', axis='y', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    return fig, ax, im

import time
from utils.eval import eval_pitchtrack

def hz_to_cents(freq, ref=1.0):
    freq = np.asarray(freq)
    # Where freq > 0, compute cents; otherwise, return NaN.
    cents = np.where(freq > 0, 1200 * np.log2(freq / ref), np.nan)
    return cents

def visualizeMultiPitchAlgo(wav, sr, gt_qfreq, 
                            algorithms=None,
                            iBlockLength=1024, iHopLength=512,
                            ref_freq=1.0):
  
    if algorithms is None:
        algorithms = ['SpectralAcf', 'SpectralHps', 'TimeAcf', 
                      'TimeAmdf', 'TimeAuditory', 'TimeZeroCrossings']
    
    results = []  # Will hold results for each algorithm
    global_min = float('inf')
    global_max = float('-inf')
    
    # Convert ground truth to numpy array (in Hz) and then to cents
    gt_qfreq = np.asarray(gt_qfreq)
    gt_cents = hz_to_cents(gt_qfreq, ref=ref_freq)
    
    for algo in algorithms:
        # Compute pitch and measure runtime
        start_time = time.time()
        est_freq, est_time = pyACA.computePitch(algo, wav, sr, 
                                                iBlockLength=iBlockLength, 
                                                iHopLength=iHopLength)
        runtime = time.time() - start_time
        
        # Compute RMS error in cents (only considers frames where both are nonzero)
        rms_error = eval_pitchtrack(estimate_in_hz=est_freq, 
                                    groundtruth_in_hz=gt_qfreq, 
                                    mode='pitch')
        
        plot_gt = np.copy(gt_qfreq)
        plot_est = np.copy(est_freq)
        mask = (gt_qfreq == 0) | (est_freq == 0)
        if mask.any():
            plot_gt[mask] = np.nan
            plot_est[mask] = np.nan
        
        plot_gt_cents = hz_to_cents(plot_gt, ref=ref_freq)
        plot_est_cents = hz_to_cents(plot_est, ref=ref_freq)
    
        current_min = np.nanmin(np.concatenate((plot_gt_cents, plot_est_cents)))
        current_max = np.nanmax(np.concatenate((plot_gt_cents, plot_est_cents)))
        global_min = min(global_min, current_min)
        global_max = max(global_max, current_max)
        
        results.append({
            'algo': algo,
            'runtime': runtime,
            'est_time': est_time,
            'plot_est_cents': plot_est_cents,
            'plot_gt_cents': plot_gt_cents,
            'rms_error': rms_error
        })
    
    if global_min == global_max:
        global_min -= 100
        global_max += 100
    
    nrows = 2
    ncols = 3
    plt.figure(figsize=(18, 10))
    
    for idx, res in enumerate(results):
        ax = plt.subplot(nrows, ncols, idx + 1)
        ax.plot(res['est_time'], res['plot_est_cents'], label='Estimated', color='b')
        ax.plot(res['est_time'], res['plot_gt_cents'], label='Ground Truth', color='r', linestyle='--')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Pitch (cents)')
        ax.set_ylim(global_min, global_max)
        ax.set_title(f"{res['algo']}\nRMS Error: {res['rms_error']:.2f} cents, Run time: {res['runtime']:.3f} s")
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()


def visualizeNoveltyFunction(d, t, peaks, target=None):
    
    # Create visualizations
    plt.figure(figsize=(20, 8))
    plt.plot(t, d)
    # put dots on the peaks
    plt.scatter(t[peaks], d[peaks], color='r', label='Estimated Onsets')
    if target is not None:
        plt.vlines(target, 0, np.max(d), colors='g', linestyles='dashed', label='Target Onsets')
    plt.xlabel('Time (s)')
    plt.ylabel('Novelty Function')
    plt.title('Novelty Function (Flux) with Detected Peaks')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    pass
