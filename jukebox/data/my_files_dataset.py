import torchaudio
import torch
import os

class FilesAudioDataset:
    def __init__(self, directory, sample_rate, min_duration, max_duration, chank_duration):
        """
        Args:
            directory (str): Path to the audio files directory.
            sample_rate (int): Desired sample rate for audio files.
            min_duration (float): Minimum duration (in seconds) of audio files.
            max_duration (float): Maximum duration (in seconds) of audio files.
            target_length (int): Number of samples for each chunk (e.g., sample_rate * desired_chunk_duration).
        """
        self.files = [os.path.join(directory,os.path.join(dir, f)) for dir in os.listdir(directory) for f in os.listdir(os.path.join(directory, dir)) if f.endswith(".wav")]
        print(f"Found {len(self.files)} audio files in {directory}")
        self.sample_rate = sample_rate
        self.min_samples = int(min_duration * sample_rate)
        # Get the durations of all audio files
        self.durations = self._calculate_durations(self.files)
        print(f"Found {len(self.durations)} audio files")
        self.max_duration_in_files = max(self.durations) if self.durations else 0  # Maximum file duration
        
        # Handle max_samples based on max_duration or maximum file duration
        if max_duration == float('inf'):
            self.max_samples = int(self.max_duration_in_files * sample_rate)
        else:
            self.max_samples = int(max_duration * sample_rate)
        self.target_length = chank_duration * sample_rate

        # Precompute chunk indices for fast access
        self.chunk_indices = self._precompute_chunk_indices()
        print(f"Precomputed {len(self.chunk_indices)} chunks")

    def _precompute_chunk_indices(self):
        """Precompute chunk indices and their corresponding file paths."""
        chunk_indices = []
        for file_idx, file in enumerate(self.files):
            num_chunks = self._num_chunks(file)
            print(f"File {file_idx}: {num_chunks} chunks")
            for chunk_idx in range(num_chunks):
                chunk_indices.append((file_idx, chunk_idx))
        return chunk_indices

    def __len__(self):
        # Return the total number of chunks
        return len(self.chunk_indices)

    def __getitem__(self, idx):
        # Use the precomputed chunk index list to quickly find the corresponding file and chunk
        file_idx, chunk_idx = self.chunk_indices[idx]
        return self._get_chunk(self.files[file_idx], chunk_idx)

    def _num_chunks(self, file):
        """Calculate the number of chunks for a given file."""
        waveform, sr = torchaudio.load(file)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)
        num_samples = waveform.shape[1]
        if num_samples < self.min_samples or num_samples > self.max_samples:
            return 0  # Ignore files outside duration range
        return (num_samples + self.target_length - 1) // self.target_length  # Ceil division

    def _get_chunk(self, file, chunk_idx):
        """Extract a specific chunk from a file."""
        waveform, sr = torchaudio.load(file)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)

        num_samples = waveform.shape[1]
        start_idx = chunk_idx * self.target_length
        end_idx = start_idx + self.target_length

        # Apply padding or truncation to extract the chunk
        if start_idx >= num_samples:
            raise IndexError("Chunk index out of range")
        chunk = waveform[:, start_idx:end_idx]
        if chunk.shape[1] < self.target_length:
            chunk = torch.nn.functional.pad(chunk, (0, self.target_length - chunk.shape[1]))

        return chunk.numpy()
    def _calculate_durations(self, files):
        durations = []
        for file in files:
            print(file)
            info = torchaudio.info(file)
            duration = info.num_frames / info.sample_rate  # Duration in seconds
            durations.append(duration)
        return durations
