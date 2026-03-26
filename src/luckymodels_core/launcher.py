from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen


def launch_runtime_process(
    root_dir: Path,
    manifest_path: str,
    profile_path: str,
    port: int,
    log_path: str | None = None,
) -> tuple[int, str]:
    env = os.environ.copy()
    env["LUCKYMODELS_MANIFEST_PATH"] = manifest_path
    env["LUCKYMODELS_PROFILE_PATH"] = profile_path

    cmd = [
        str(root_dir / ".venv" / "bin" / "python"),
        "-m",
        "uvicorn",
        "runtime.server:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    stdout_target = subprocess.DEVNULL
    stderr_target = subprocess.DEVNULL
    log_handle = None
    if log_path:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_file.open("a", encoding="utf-8")
        log_handle.write(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] launching runtime "
            f"for {Path(manifest_path).parent.name} on 127.0.0.1:{port}\n"
        )
        log_handle.flush()
        stdout_target = log_handle
        stderr_target = log_handle

    process = subprocess.Popen(  # noqa: S603
        cmd,
        cwd=root_dir,
        env=env,
        stdout=stdout_target,
        stderr=stderr_target,
        start_new_session=True,
    )
    if log_handle is not None:
        log_handle.close()
    return process.pid, f"http://127.0.0.1:{port}"


def wait_for_runtime(
    service_url: str,
    timeout_seconds: float = 10.0,
    interval_seconds: float = 0.5,
) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urlopen(service_url + "/healthz", timeout=2) as response:  # noqa: S310
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(interval_seconds)
    return False


def process_is_running(pid: int | None) -> bool:
    if not pid:
        return False
    state = _process_state(pid)
    if state == "Z":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_runtime_process(pid: int | None, timeout_seconds: float = 5.0) -> bool:
    if not pid:
        return False

    try:
        process_group_id = os.getpgid(pid)
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return True

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not process_is_running(pid):
            return True
        time.sleep(0.2)

    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return True

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if not process_is_running(pid):
            return True
        time.sleep(0.1)
    return not process_is_running(pid)


def tail_log(log_path: str | None, lines: int = 100) -> str:
    if not log_path:
        return ""
    path = Path(log_path)
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8") as handle:
        content = handle.readlines()
    return "".join(content[-max(1, lines) :])


def _process_state(pid: int) -> str | None:
    stat_path = Path("/proc") / str(pid) / "stat"
    if not stat_path.exists():
        return None
    try:
        return stat_path.read_text(encoding="utf-8").split()[2]
    except Exception:
        return None
