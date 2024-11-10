import os
import json

class ProgressTracker:
    def __init__(self, progress_file):
        self.progress_file = progress_file
        self.stage = ''
        self.processed_batches = 0
        self.total_batches = 0
        self.load_progress()

    def load_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.stage = data.get('stage', '')
                    self.processed_batches = data.get('processed_batches', 0)
                    self.total_batches = data.get('total_batches', 0)
            except (json.JSONDecodeError, ValueError):
                # Handle the case where the file is empty or contains invalid JSON
                print("Progress file is empty or invalid. Resetting progress.")
                self.reset_progress()
        else:
            self.reset_progress()

    def save_progress(self):
        data = {
            'stage': self.stage,
            'processed_batches': self.processed_batches,
            'total_batches': self.total_batches
        }
        with open(self.progress_file, 'w') as f:
            json.dump(data, f)

    def update_stage(self, stage):
        self.stage = stage
        self.save_progress()

    def update_batches(self, processed, total):
        self.processed_batches = processed
        self.total_batches = total
        self.save_progress()

    def reset_progress(self):
        self.stage = ''
        self.processed_batches = 0
        self.total_batches = 0
        self.save_progress()

    def display_progress(self):
        print("\nCurrent Progress:")
        print(f"Stage: {self.stage}")
        print(f"Processed Batches: {self.processed_batches}/{self.total_batches}")
