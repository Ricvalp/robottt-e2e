import os
import torch
from torch.profiler import ProfilerActivity, profile, schedule


class ProfilerGuard:
    def __init__(self, use: bool, start_step: int, end_step: int, trace_path: str):
        self.use = use
        self.start_step = start_step
        self.end_step = end_step
        self.trace_path = trace_path
        trace_dir = os.path.dirname(trace_path)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)

        acts = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            acts.append(ProfilerActivity.CUDA)
        self.activities = acts

        self._prof = None

    def start(self, current_step: int):
        """Start profiling window. Call once when you hit start_step."""
        if not self.use or self._prof is not None:
            return
        if current_step != self.start_step:
            return
        # active steps is inclusive range length; caller must call step() once per iter
        n_active = self.end_step - self.start_step + 1
        if n_active <= 0:
            raise ValueError("end_step must be >= start_step")

        self._prof = profile(
            activities=self.activities,
            schedule=schedule(wait=0, warmup=0, active=n_active, repeat=1),
            record_shapes=False,
            with_stack=False,
        )
        self._prof.__enter__()
        # account for the iteration in which start() is called
        self._prof.step()

    def step(self):
        """Advance profiler by one training iteration. Call every iteration while active."""
        if self._prof is not None:
            self._prof.step()

    def stop(self, current_step: int):
        """Stop and export. Call once after the last profiled step."""
        if self._prof is None:
            return
        if current_step < self.end_step:
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._prof.export_chrome_trace(self.trace_path)
            print(f"Saved profiling trace to {self.trace_path}")
        finally:
            self._prof.__exit__(None, None, None)
            self._prof = None
