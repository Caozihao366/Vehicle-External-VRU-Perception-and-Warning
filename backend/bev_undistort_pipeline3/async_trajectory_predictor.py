"""Run QCNet trajectory prediction outside the BEV processing thread."""

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Optional

from bev_types import BEVFrame


_STOP = object()
_RESET = object()


@dataclass(frozen=True)
class PredictionPacket:
    """A prediction together with the BEV frame that produced it."""

    frame_id: int
    timestamp: float
    result: object


@dataclass(frozen=True)
class PredictionRequest:
    """One immutable model input prepared by the source-time history buffer."""

    frame_id: int
    timestamp: float
    model_input: dict
    target_ids: tuple[str, ...]


class _RealtimeTrajectoryPredictor:
    """Own the QCNet provider and execute model forward calls."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str,
    ) -> None:
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
    def update(self, request: PredictionRequest):
        if request.model_input.get("coordinate_frame") != "ego":
            raise ValueError("QCNet requires coordinate_frame='ego'.")
        return self._provider.predict(
            request.model_input,
            target_ids=request.target_ids,
        )

    def close(self) -> None:
        self._provider.close()

class AsyncTrajectoryPredictor:
    """Run QCNet in a dedicated worker so BEV frame processing stays responsive."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device

        self._input_queue = Queue(maxsize=1)
        self._result_queue = Queue(maxsize=1)
        self._error_queue = Queue(maxsize=1)

        self._ready = Event()
        self._startup_error: Optional[BaseException] = None
        self._thread = Thread(
            target=self._run,
            name="qcnet-prediction-thread",
            daemon=True,
        )

    def start(self) -> None:
        """Start the worker and wait only for the one-time model load."""

        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise RuntimeError(
                f"Failed to start QCNet predictor: {self._startup_error}"
            ) from self._startup_error

    def submit(self, frame: BEVFrame, model_input: dict, target_ids) -> None:
        """Enqueue the newest prepared input without running forward here."""

        if not isinstance(frame, BEVFrame):
            raise TypeError("submit() requires a BEVFrame.")
        if not self._thread.is_alive():
            raise RuntimeError("QCNet prediction thread is not running.")
        request = PredictionRequest(
            frame_id=int(frame.frame_id),
            timestamp=float(frame.timestamp),
            model_input=model_input,
            target_ids=tuple(str(item) for item in target_ids),
        )
        self._replace_latest(self._input_queue, request)

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
            self._replace_latest(self._input_queue, _RESET)

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
                    self._replace_latest(self._result_queue, None)
                    continue

                try:
                    prediction = predictor.update(item)
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
