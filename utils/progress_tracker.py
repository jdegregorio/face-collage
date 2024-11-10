import os
import json

class ProgressTracker:
    def __init__(self, progress_file):
        self.progress_file = progress_file
        self.stage = ''
        self.load_progress()

    def load_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.stage = data.get('stage', '')
            except (json.JSONDecodeError, ValueError):
                print("Progress file is empty or invalid. Resetting progress.")
                self.reset_progress()
        else:
            self.reset_progress()

    def save_progress(self):
        data = {'stage': self.stage}
        with open(self.progress_file, 'w') as f:
            json.dump(data, f)

    def update_stage(self, stage):
        self.stage = stage
        self.save_progress()

    def reset_progress(self):
        self.stage = ''
        self.save_progress()

    def display_progress(self):
        print("\nCurrent Progress:")
        print(f"Stage: {self.stage}")
