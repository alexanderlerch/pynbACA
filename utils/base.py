import timeit
import numpy as np
from scipy.signal import sawtooth
from pyACA import ToolFreq2Midi
#TODO: use consistent function names, pyACA style would be compareRuntime, generateSinWave
def compare_runtime(func1, func2, input_sequence, num_runs=10):
    """
    Compare the execution time of two functions with the same input.
    
    :param func1: First function to test
    :param func2: Second function to test
    :param input_sequence: The input sequence (1D array) to be used in both functions
    :param num_runs: Number of runs to average the execution time
    :return: Dictionary with execution times for both functions
    """
    
    # Define wrapper functions to measure time
    def wrapper_1():
        func1(np.copy(input_sequence))  # Copy to avoid side effects
    
    def wrapper_2():
        func2(np.copy(input_sequence))

    # Measure execution time using timeit (best of num_runs runs)
    time_func1 = timeit.timeit(wrapper_1, number=num_runs) / num_runs
    time_func2 = timeit.timeit(wrapper_2, number=num_runs) / num_runs

    # Print results
    # print(f"Average execution time of {func1.__name__}: {time_func1:.6f} sec")
    # print(f"Average execution time of {func2.__name__}: {time_func2:.6f} sec")
    
    return time_func1, time_func2


#TODO: be sure to use consistent parameter names (see frequency vs. sr), also preferably add unit, e.g., freqInHz, durationInS, etc.
def sineWavGen(frequency=440, duration=2.0, sr=44100, iBlockLength=1024, iHopLength=512):
    # Create time vector for the entire signal
    t = np.linspace(0, duration, int(round(sr * duration)), endpoint=False)
    wav = np.sin(2 * np.pi * frequency * t)
    
    # Compute the number of blocks (frames) that will be used in pitch analysis.
    n_frames = int(np.floor((len(wav)) / iHopLength)) + 1
    # Create a constant ground truth pitch array.
    gt_qfreq = np.full(n_frames, frequency)
    
    return wav, sr, gt_qfreq

def sawtoothWavGen(frequency=440, duration=2.0, sr=44100, iBlockLength=1024, iHopLength=512):
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    wav = sawtooth(2 * np.pi * frequency * t)
    n_frames = int(np.floor(n_samples / iHopLength)) + 1
    gt_qfreq = np.full(n_frames, frequency)
    return wav, sr, gt_qfreq

def gen_chromatic_scale(start_midi=69, n_semitones=12, seconds_per_note=1.0,
                         sr=44100, iBlockLength=4096, iHopLength=512):
    """
    Build a chromatic scale by concatenating single-note sines.
    Returns: wav_all, sr, gt_qfreq_all, gt_midi_all
    """
    wavs = []
    
    for k in range(n_semitones):
        midi = start_midi + k
        f0 = 440.0 * (2.0 ** ((midi - 69.0)/12.0))
        w, sr_out, gt_f = sineWavGen(f0, seconds_per_note, sr=sr,
                                     iBlockLength=iBlockLength, iHopLength=iHopLength)
        if sr_out != sr:
            raise ValueError("Sample-rate mismatch in generator.")
        wavs.append(w)
    
    wav_all = np.concatenate(wavs, axis=0)
    
    # Let pyACA determine the frame count by running a dummy pitch estimation
    import pyACA
    dummy_pitch, dummy_time = pyACA.computePitch('TimeAcf', wav_all, sr, 
                                                   iBlockLength=iBlockLength, 
                                                   iHopLength=iHopLength)
    n_frames_total = len(dummy_pitch)
    
    # Now build ground truth with the correct frame count
    samples_per_note = int(seconds_per_note * sr)
    gt_qfreq_all = np.zeros(n_frames_total)
    
    for i in range(n_frames_total):
        center_sample = i * iHopLength + iBlockLength // 2
        note_idx = min(center_sample // samples_per_note, n_semitones - 1)
        midi = start_midi + note_idx
        gt_qfreq_all[i] = 440.0 * (2.0 ** ((midi - 69.0)/12.0))
    
    gt_midi_all = ToolFreq2Midi(gt_qfreq_all)
    
    return wav_all, sr, gt_qfreq_all, gt_midi_all

def gen_chromatic_scale_saw(start_midi=69, n_semitones=12, seconds_per_note=1.0,
                         sr=44100, iBlockLength=4096, iHopLength=512):
    """
    Build a chromatic scale by concatenating single-note sines.
    Returns: wav_all, sr, gt_qfreq_all, gt_midi_all
    """
    wavs = []
    
    for k in range(n_semitones):
        midi = start_midi + k
        f0 = 440.0 * (2.0 ** ((midi - 69.0)/12.0))
        w, sr_out, gt_f = sawtoothWavGen(f0, seconds_per_note, sr=sr,
                                     iBlockLength=iBlockLength, iHopLength=iHopLength)
        if sr_out != sr:
            raise ValueError("Sample-rate mismatch in generator.")
        wavs.append(w)
    
    wav_all = np.concatenate(wavs, axis=0)
    
    # Let pyACA determine the frame count by running a dummy pitch estimation
    import pyACA
    dummy_pitch, dummy_time = pyACA.computePitch('TimeAcf', wav_all, sr, 
                                                   iBlockLength=iBlockLength, 
                                                   iHopLength=iHopLength)
    n_frames_total = len(dummy_pitch)
    
    # Now build ground truth with the correct frame count
    samples_per_note = int(seconds_per_note * sr)
    gt_qfreq_all = np.zeros(n_frames_total)
    
    for i in range(n_frames_total):
        center_sample = i * iHopLength + iBlockLength // 2
        note_idx = min(center_sample // samples_per_note, n_semitones - 1)
        midi = start_midi + note_idx
        gt_qfreq_all[i] = 440.0 * (2.0 ** ((midi - 69.0)/12.0))
    
    gt_midi_all = ToolFreq2Midi(gt_qfreq_all)
    
    return wav_all, sr, gt_qfreq_all, gt_midi_all

def tickGen(bpm, duration_beats=8, sample_rate=44100, tick_duration=0.1, frequency=1000):

    beat_duration = 60.0 / bpm
    
    if tick_duration > beat_duration:
        raise ValueError("tick_duration cannot be greater than the beat duration (60/BPM).")
    
    # Number of samples for tick sound and silence
    tick_samples = int(sample_rate * tick_duration)
    silence_samples = int(sample_rate * (beat_duration - tick_duration))
    
    t = np.linspace(0, tick_duration, tick_samples, endpoint=False)
    tick_wave = np.sin(2 * np.pi * frequency * t)
    decay = np.linspace(1, 0, tick_samples)
    tick_wave *= decay

    silence_wave = np.zeros(silence_samples)
    one_beat = np.concatenate([tick_wave, silence_wave])
    
    audio_signal = np.tile(one_beat, duration_beats)
    
    # Normalize the signal to the range of int16
    if np.max(np.abs(audio_signal)) != 0:
        audio_signal = audio_signal / np.max(np.abs(audio_signal))
    audio_signal_int16 = np.int16(audio_signal * 32767)
    
    return sample_rate, audio_signal_int16