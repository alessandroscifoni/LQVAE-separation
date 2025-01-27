import random
from torch.utils.data import DataLoader, random_split
from jukebox.data.my_files_dataset import FilesAudioDataset
class DataProcessor:
    def __init__(self, train_dir, test_dir, sample_rate=16000, batch_size=16, min_duration=1.0, max_duration=float('inf')):
        """
        Args:
            train_dir (str): Path to the training dataset directory.
            test_dir (str): Path to the testing dataset directory.
            sample_rate (int): Sample rate for audio processing.
            batch_size (int): Batch size for DataLoader.
            min_duration (float): Minimum duration for audio files.
            max_duration (float): Maximum duration for audio files.
        """
        # Create datasets
        self.train_dataset = FilesAudioDataset(train_dir, sample_rate, min_duration, max_duration)
        self.test_dataset = FilesAudioDataset(test_dir, sample_rate, min_duration, max_duration)
        # print(f"training dataset len len {self.train_dataset}")
        # Split train dataset for train/test splits if needed
        total_samples = len(self.train_dataset)
        test_size = int(0.1 * total_samples)  # 10% for test
        train_size = total_samples - test_size

        self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_size, test_size])
        # print(f"training dataset len len {self.train_dataset}")
        # Create DataLoaders
        self.batch_size = batch_size
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        self.print_stats()

    def print_stats(self):
        print(f"Train Dataset: {len(self.train_dataset)} samples")
        print(f"Validation Dataset: {len(self.val_dataset)} samples")
        print(f"Test Dataset: {len(self.test_dataset)} samples")
        print(f"Batch Size: {self.batch_size}")
