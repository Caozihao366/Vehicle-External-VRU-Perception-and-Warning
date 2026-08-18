"""Run QCNet trajectory prediction outside the BEV processing thread."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Dict, Optional

import numpy as np

from bev_types import BEVFrame


_STOP = object()
_RESET = object()


@dataclass(frozen=True)
class PredictionPacket:
    """A prediction together with the BEV frame that produced it."""

    frame_id: int
    timestamp: float
    result: object


class _RealtimeTrajectoryPredictor:
    """Adapt streaming ``BEVFrame`` objects to the existing QCNet API."""

    history_steps = 50

    def __init__(self, checkpoint_path: str, device: str) -> None:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

        # The provider is deployment-only. It loads the model once and also
        # converts raw QCNet tensors to the multimodal format consumed by TTC.
        from ttc.integrations.qcnet import QCNetTrajectoryProvider

        self._provider = QCNetTrajectoryProvider(
            package_root=checkpoint.parent.parent,
            device=device,
        )
        self._frames = deque(maxlen=self.history_steps)

    @property
    def history_frames(self) -> int:
        return len(self._frames)

    def update(self, frame: BEVFrame):
        if frame.coordinate_frame != "ego":
            raise ValueError("QCNet requires coordinate_frame='ego'.")

        if self._frames and frame.timestamp <= self._frames[-1].timestamp:
            raise ValueError("BEVFrame timestamps must be strictly increasing.")

        self._frames.append(frame)
        model_input, target_ids = self._build_model_input()
        if model_input is None or not target_ids:
            return None

        return self._provider.predict(model_input, target_ids=target_ids)

    def reset_history(self) -> None:
        self._frames.clear()

    def close(self) -> None:
        self._provider.close()

    @staticmethod
    def _agent_type(type_name: str) -> int:
        return {
            "vehicle": 0,
            "pedestrian": 1,
            "motorcyclist": 2,
            "cyclist": 3,
            "bus": 4,
        }.get(type_name, 9)

    def _build_model_input(self):
        # QCNet needs two valid samples for every agent predicted at t0. The
        # remaining leading history is represented by an invalid-mask pad.
        if len(self._frames) < 2:
            return None, []

        latest = self._frames[-1]
        occurrence_count: Dict[str, int] = {}
        for frame in self._frames:
            for obj in frame.objects:
                agent_id = str(obj.id)
                occurrence_count[agent_id] = occurrence_count.get(agent_id, 0) + 1

        target_ids = []
        seen = set()
        for obj in latest.objects:
            agent_id = str(obj.id)
            if occurrence_count.get(agent_id, 0) >= 2 and agent_id not in seen:
                target_ids.append(agent_id)
                seen.add(agent_id)

        if not target_ids:
            return None, []

        agent_ids = ["ego"] + target_ids
        index_by_id = {
            agent_id: index for index, agent_id in enumerate(agent_ids)
        }
        num_agents = len(agent_ids)

        position = np.zeros(
            (num_agents, self.history_steps, 2), dtype=np.float32
        )
        velocity = np.zeros_like(position)
        heading = np.zeros(
            (num_agents, self.history_steps), dtype=np.float32
        )
        valid_mask = np.zeros(
            (num_agents, self.history_steps), dtype=bool
        )
        agent_type = np.zeros((num_agents,), dtype=np.int64)

        start_step = self.history_steps - len(self._frames)
        for frame_offset, frame in enumerate(self._frames):
            step = start_step + frame_offset
            position[0, step] = frame.ego.position
            velocity[0, step] = frame.ego.velocity
            heading[0, step] = frame.ego.heading
            valid_mask[0, step] = True

            for obj in frame.objects:
                index = index_by_id.get(str(obj.id))
                if index is None:
                    continue
                position[index, step] = obj.position
                velocity[index, step] = obj.velocity
                heading[index, step] = obj.heading
                valid_mask[index, step] = True
                agent_type[index] = self._agent_type(obj.type)

        return {
            "position": position,
            "velocity": velocity,
            "heading": heading,
            "valid_mask": valid_mask,
            "agent_type": agent_type,
            "av_index": 0,
            "coordinate_frame": "ego",
            "frequency_hz": 10.0,
            "agent_ids": agent_ids,
        }, target_ids


class AsyncTrajectoryPredictor:
    """Run QCNet in a dedicated worker so BEV frame processing stays responsive."""

    def __init__(self, checkpoint_path: str, device: str = "cuda") -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device

        self._input_queue = Queue()
        self._result_queue = Queue(maxsize=1)
        self._error_queue = Queue(maxsize=1)

        self._ready = Event()
        self._startup_error: Optional[BaseException] = None
        self._history_frames = 0
        self._thread = Thread(
            target=self._run,
            name="qcnet-prediction-thread",
            daemon=True,
        )

    @property
    def history_frames(self) -> int:
        return self._history_frames

    def start(self) -> None:
        """Start the worker and wait only for the one-time model load."""

        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise RuntimeError(
                f"Failed to start QCNet predictor: {self._startup_error}"
            ) from self._startup_error

    def submit(self, frame: BEVFrame) -> None:
        """Enqueue one frame without running model forward in this thread."""

        if not isinstance(frame, BEVFrame):
            raise TypeError("submit() requires a BEVFrame.")
        if not self._thread.is_alive():
            raise RuntimeError("QCNet prediction thread is not running.")
        self._input_queue.put_nowait(frame)

    def get_latest(self) -> Optional[PredictionPacket]:
        """Return the newest completed prediction without blocking."""

        latest = None
        try:
            while True:
                latest = self._result_queue.get_nowait()
        except Empty:
            return latest

    def get_latest_error(self) -> Optional[BaseException]:
        """Return the newest worker error without blocking."""

        latest = None
        try:
            while True:
                latest = self._error_queue.get_nowait()
        except Empty:
            return latest

    def reset_history(self) -> None:
        """Clear scenario history after localization, sequence, or MOT reset."""

        if self._thread.is_alive():
            self._input_queue.put_nowait(_RESET)

    def stop(self) -> None:
        """Stop the worker and release the model."""

        if not self._thread.is_alive():
            return

        # Pending frames no longer matter during shutdown. Removing them lets
        # the worker observe the stop marker immediately after an active forward.
        try:
            while True:
                self._input_queue.get_nowait()
        except Empty:
            pass
        self._input_queue.put_nowait(_STOP)
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            raise RuntimeError("QCNet prediction thread did not stop in time.")

    @staticmethod
    def _replace_latest(queue: Queue, value) -> None:
        try:
            while True:
                queue.get_nowait()
        except Empty:
            pass
        queue.put_nowait(value)

    def _run(self) -> None:
        predictor = None
        try:
            predictor = _RealtimeTrajectoryPredictor(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
            )
            self._ready.set()

            while True:
                item = self._input_queue.get()
                if item is _STOP:
                    break
                if item is _RESET:
                    predictor.reset_history()
                    self._history_frames = 0
                    continue

                try:
                    prediction = predictor.update(item)
                    self._history_frames = predictor.history_frames
                    if prediction is None:
                        continue
                    self._replace_latest(
                        self._result_queue,
                        PredictionPacket(
                            frame_id=item.frame_id,
                            timestamp=item.timestamp,
                            result=prediction,
                        ),
                    )
                except (ValueError, RuntimeError) as error:
                    self._replace_latest(self._error_queue, error)
        except BaseException as error:
            if not self._ready.is_set():
                self._startup_error = error
            else:
                self._replace_latest(self._error_queue, error)
            self._ready.set()
        finally:
            if predictor is not None:
                predictor.close()
            self._ready.set()
