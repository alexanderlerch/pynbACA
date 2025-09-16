## Repository Structure

The repository is organized into two main parts:

- **`utils/`**  
  A collection of reusable utility functions that support the notebooks.

- **`Notebooks/`**  
  Contains Jupyter notebooks for different topics.  
  Each notebook is named using the convention: `\<number\>-\<topic\>.ipynb`
  For example:
  - `01_DSP_Fundamental_Concepts.ipynb`
  - `02_NoveltyOnset.ipynb`

### Audio Asset Directory Structure

Since currently the audio asset repository is not created yet,I just put the Audio_Asset directory in this branch. In future, it will be moved to a independent repository and linked to this repo.

Audio assets are organized by the notebook they belong to, so that each notebook has its own dedicated subdirectory.  
The naming convention is: Audio_Asset/`<number>-<notebook_name>/<number>-<audio_file_name>.wav`
For example: - `Audio_Asset/01-DSP_Fundamental_Concepts/01-D_AMairena.wav`

## `01_DSP_Fundamental_Concepts.ipynb`

### Update

- [x] Add the dependency installation part in environment setup, remove the outdated/unused dependency
- [x] Add a brief introduction section before the Section 0
- [x] Renamed variable wav as wav_audio to reduce potential conflict with the imported library "scipy.io.wav as wav"
- [x] Added the missing equation of DFT, FFT, STFT
- [x] Modified audio file path structure
- [x] Clear the specific warnings.
