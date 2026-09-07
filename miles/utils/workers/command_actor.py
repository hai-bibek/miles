import logging
import os
import signal
import subprocess
import threading

from miles.utils.misc import NodeProbeMixin
from miles.utils.test_utils import fault_injector
from miles.utils.workers import process_utils

logger = logging.getLogger(__name__)


class CommandActor(NodeProbeMixin):
    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._shutting_down = False

    def run(self, cmd: str, envs: dict[str, str]) -> None:
        assert self._process is None, "CommandActor.run can only be called once"

        logger.info(f"CommandActor launches subprocess cmd={cmd!r} env_names={sorted(envs)}")
        self._process = process_utils.launch_bound_subprocess(["/bin/sh", "-c", cmd], envs=envs)

        threading.Thread(target=self._babysit, args=(self._process,), daemon=True).start()

    def shutdown(self) -> None:
        if self._process is None:
            return

        self._shutting_down = True
        process_utils.terminate_process_tree(self._process)

    def kill_subprocess(self) -> None:
        assert self._process is not None, "CommandActor has no subprocess to kill"
        process_utils.kill_process_tree(self._process)

    def inject_fault(self, mode: str) -> None:
        assert self._process is not None, "CommandActor has no subprocess to inject a fault into"
        failure_mode = fault_injector.FailureMode(mode)
        signals = {
            fault_injector.FailureMode.SIGKILL: signal.SIGKILL,
            fault_injector.FailureMode.SIGSTOP: signal.SIGSTOP,
        }
        if failure_mode not in signals:
            raise ValueError(f"Unsupported subprocess fault: {failure_mode.value}")

        logger.warning(f"CommandActor sends {failure_mode.value} to subprocess group pid={self._process.pid}")
        os.killpg(self._process.pid, signals[failure_mode])

    def _babysit(self, process: subprocess.Popen) -> None:
        returncode = process.wait()

        if self._shutting_down:
            logger.info(f"CommandActor subprocess exited with returncode={returncode} during shutdown")
            return

        logger.info(f"CommandActor exits since its subprocess exited with returncode={returncode}")
        os._exit(returncode if 0 <= returncode <= 255 else 1)
