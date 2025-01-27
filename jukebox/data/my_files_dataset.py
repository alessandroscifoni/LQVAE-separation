import torchaudio
from torch.utils.data import Dataset
import librosa

class FilesAudioDataset(Dataset):
    def __init__(self, audio_files_dir, sample_rate=16000, min_duration=1.0, max_duration=float('inf')):
        """
        Args:
            audio_files_dir (str): Path to the directory containing audio files.
            sample_rate (int): Target sample rate for audio files.
            min_duration (float): Minimum duration (in seconds) to include a file.
            max_duration (float): Maximum duration (in seconds) to include a file.
        """
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration

        # Find all audio files in the directory
        self.files = librosa.util.find_files(audio_files_dir, ext=['wav'])
        self.durations = []

        # Filter files based on duration
        for file in self.files:
            info = torchaudio.info(file)
            duration = info.num_frames / info.sample_rate
            if min_duration <= duration <= max_duration:
                self.durations.append(duration)
            else:
                self.files.remove(file)

        print(f"Filtered {len(self.files)} files from {audio_files_dir} based on duration.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load the audio file
        file_path = self.files[idx]
        waveform, sr = torchaudio.load(file_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resample(waveform)

        return waveform.numpy()  # Return waveform and file path for debugging/metadata
