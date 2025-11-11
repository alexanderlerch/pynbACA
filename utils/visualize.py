import matplotlib.pyplot as plt
import numpy as np
import pyACA
import matplotlib.animation as animation
import pretty_midi
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

def visualizeSpec(stft, sr, hop_length=512, log_magnitude=True, ax=None,
                  fig_width=10, fig_height=6):

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
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
    img = ax.imshow(magnitude_to_plot, aspect='auto', origin='lower',
                    cmap='viridis', extent=extent)
    plt.colorbar(img, ax=ax, label=colorbar_label)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    return ax

'''
def visualizePitchTracking(gt_time, gt_midi, est_time=None, est_midi=None):

    plt.figure(figsize=(10, 6))
    plt.plot(gt_time, gt_midi, linestyle='-', color='b', label='Ground Truth')
    if est_midi is not None and est_time is not None:
        plt.plot(est_time, est_midi, linestyle='--', color='r', label='Estimated')
    plt.xlabel('Time (s)')
    plt.ylabel('Midi note')
    plt.title('Pitch Tracking Visualization')
    plt.grid(True)
    plt.legend()
    plt.show()
'''

def visualizePitchTracking(gt_time, gt_midi, est_time=None, est_midi=None, algorithm_name=''):
    gt_time = np.asarray(gt_time)
    gt_midi = np.asarray(gt_midi)

    plt.figure(figsize=(10, 6))

    # plot Ground Truth with horizontal lines only (skip zeros) 
    # find boundaries where the MIDI value changes
    boundaries = np.where(np.diff(gt_midi) != 0)[0] + 1
    starts = np.concatenate(([0], boundaries))
    ends   = np.concatenate((boundaries, [len(gt_midi)]))

    first_label_used = False
    for s, e in zip(starts, ends):
        val = gt_midi[s]
        if val > 0:  # draw only active notes
            plt.hlines(y=val,xmin=gt_time[s],xmax=gt_time[e-1],color='b',linewidth=2,label='Ground Truth' if not first_label_used else None)
            first_label_used = True

    if est_midi is not None and est_time is not None:
        plt.plot(est_time, est_midi, linestyle='--', color='r', label='Estimated')

    plt.xlabel('Time (s)')
    plt.ylabel('Midi note')
    plt.title(f'Pitch Tracking Visualization - {algorithm_name}')
    plt.grid(True)
    plt.legend()
    plt.show()

def visualize_tracking_freq(est_freq, gd_freq, freq_rms, title):
    """
    Plot Estimated vs Ground-Truth frequency in Hz (vs frame index).
    Unvoiced (<=0) are hidden. Title shows RMS (Hz).
    """
    est = np.asarray(est_freq, dtype=float)
    gt  = np.asarray(gd_freq,   dtype=float)
    L = min(len(est), len(gt))
    est, gt = est[:L], gt[:L]

    # Hide unvoiced
    mask = (est <= 0) | (gt <= 0) | ~np.isfinite(est) | ~np.isfinite(gt)
    est_plot = est.copy()
    gt_plot  = gt.copy()
    est_plot[mask] = np.nan
    gt_plot[mask]  = np.nan

    x = np.arange(L)
    plt.figure(figsize=(8, 4))
    plt.plot(x, gt_plot,  '-',  label='Ground Truth (Hz)')
    plt.plot(x, est_plot, '--', label='Estimated (Hz)')
    plt.xlabel('Frame')
    plt.ylabel('Frequency (Hz)')
    plt.title(f"{title} — RMS Error: {freq_rms:.2f} Hz")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_tracking_midi(est_midi, gt_midi, midi_rms, title):
    """
    Plot Estimated vs Ground-Truth MIDI (vs frame index).
    Unvoiced (NaN) are hidden. Title shows RMS (cents).
    """
    est = np.asarray(est_midi, dtype=float)
    gt  = np.asarray(gt_midi,  dtype=float)
    L = min(len(est), len(gt))
    est, gt = est[:L], gt[:L]

    # Hide invalids (NaN/inf)
    mask = ~np.isfinite(est) | ~np.isfinite(gt)
    est_plot = est.copy()
    gt_plot  = gt.copy()
    est_plot[mask] = np.nan
    gt_plot[mask]  = np.nan

    x = np.arange(L)
    plt.figure(figsize=(8, 4))
    plt.plot(x, gt_plot,  '-',  label='Ground Truth (MIDI)')
    plt.plot(x, est_plot, '--', label='Estimated (MIDI)')
    plt.xlabel('Frame')
    plt.ylabel('MIDI Note Number')
    plt.title(f"{title} — RMS Error: {midi_rms:.2f} cents")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
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


def visualize_chromagram(chroma, title="Chromagram"):
    plt.figure(figsize=(10, 6))
    plt.imshow(chroma, aspect='auto', origin='lower', interpolation='nearest')
    plt.colorbar(label="Normalized Energy")

    plt.yticks(
        ticks=np.arange(12),
        labels=["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    )
    plt.xlabel("Time Frame")
    plt.ylabel("Pitch Class")
    plt.title(title)
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

# draw out the spectrogram of audio with fundamental frequency
def visualizeRefSpec(stft, sr, hop_length=512, log_magnitude=True, ax=None, ref_f = None,fig_width=10, fig_height=6):
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
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    img = ax.imshow(magnitude_to_plot, aspect='auto', origin='lower',
                    cmap='viridis', extent=extent)
    plt.colorbar(img, ax=ax, label=colorbar_label)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    if ref_f is not None:
      ax.axhline(y=ref_f, color="red", linestyle="--", linewidth=1.2)
      ax.text(time_axis[-1], ref_f, f' {ref_f:.2f} Hz', color="red", va='bottom', ha='left', fontsize=9)
    return ax

def midi_note_mask(pm, frame_times):
    """
    Return a boolean mask of len(frame_times), 
    True means MIDI note is active at that time.
    """
    ft = np.asarray(frame_times, dtype=float)
    mask = np.zeros_like(ft, dtype=bool)
    for inst in pm.instruments:
        for n in inst.notes:
            mask |= (ft >= n.start) & (ft < n.end)
    return mask


def apply_mask_to_tracks(est_freq, gt_freq, mask):
    est = np.asarray(est_freq, dtype=float).copy()
    gt  = np.asarray(gt_freq,  dtype=float).copy()
    # Hide frames outside notes OR where gt has no pitch (0)
    keep = mask & (gt > 0)
    est[~keep] = np.nan
    gt[~keep]  = np.nan
    return est, gt, keep

def midi_to_array(midi_obj):
    """Convert PrettyMIDI object → arrays of note (pitch, start, end)."""
    pitches = []
    starts = []
    ends = []
    for inst in midi_obj.instruments:
        for n in inst.notes:
            pitches.append(n.pitch)
            starts.append(n.start)
            ends.append(n.end)
    return np.array(pitches), np.array(starts), np.array(ends)

    
if __name__ == '__main__':
    pass


