#!/usr/bin/env python3
"""One official BF16 TP2 replica with resumable, state-local annotation work."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

from kd_runtime import ROOT, SOURCE_ROOT, atomic_json
from annotation_queue import AnnotationQueue

MODEL = Path("/tmp/jev-kd-20261101/models/KnowLine-31B-Gen1")
CODE = SOURCE_ROOT / "staged/scale-series-20261102/code"
CACHE = ROOT / "outputs/teacher"
processes = []
handles = []


def launch(command, name, devices=None):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    env["PATH"] = str(Path(sys.executable).parent) + ":" + env.get("PATH", "")
    if devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = devices
    cuda = Path("/data/apps/cuda/13.0")
    if cuda.exists():
        env["CUDA_HOME"] = str(cuda)
        env["PATH"] = str(cuda / "bin") + ":" + env.get("PATH", "")
    handle = (ROOT / "logs" / f"{name}.log").open("a", buffering=1)
    handles.append(handle)
    process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    processes.append(process)
    return process


def wait_ready(port, process, timeout=1800):
    started = time.monotonic()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"Teacher service on {port} exited with {process.returncode}; see its log")
        try:
            with opener.open(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(5)
    raise RuntimeError(f"Teacher service startup timed out on {port}")


def publish_progress(queue):
    value = queue.progress()
    atomic_json(CACHE / "progress.json", value)
    return value


def complete(queue):
    state = queue.snapshot()
    if not queue.complete(state):
        raise RuntimeError("Annotation ownership or committed full-pool coverage is incomplete")
    manifest = json.loads((CACHE / "manifest.json").read_text())
    timing = queue.timing()
    now = time.time()
    elapsed = now-(timing["first_annotation_started_unix"] or now)
    newly_written = manifest["pairs"]-timing["first_recovered_pairs"]
    value = dict(complete=True, pairs=manifest["pairs"], domains=manifest["domains"],
                 annotation_seconds=elapsed, total_stage_seconds=now-timing["first_stage_started_unix"],
                 pairs_per_second=newly_written/max(1, elapsed), replicas=1, tensor_parallel=2,
                 scheduling=state["mode"], committed_tasks=state["completed_tasks"],
                 recovered_pairs_this_run=state["recovered_pairs"],
                 newly_written_pairs_this_run=state["cached_pairs"]-state["recovered_pairs"],
                 newly_written_pairs_total=newly_written,
                 newly_written_pairs_per_second=newly_written/max(1,elapsed))
    atomic_json(CACHE / "complete.json", value)
    publish_progress(queue)
    print(json.dumps({"event": "teacher_annotation_complete", **value}), flush=True)


def main():
    if (CACHE / "complete.json").exists():
        print("Teacher cache already complete", flush=True)
        return
    if not (MODEL / "DOWNLOAD_COMPLETE.json").exists():
        raise RuntimeError("Complete BF16 Teacher download is not ready")
    started = time.monotonic()
    subprocess.run([sys.executable, str(CODE / "annotate_teacher.py"), "--prepare"], check=True)
    queue = AnnotationQueue(CACHE)
    timing = queue.timing()
    if "first_recovered_pairs" not in timing:
        timing["first_recovered_pairs"] = queue.snapshot()["recovered_pairs"]
        atomic_json(CACHE/"timing.json", timing)
    publish_progress(queue)
    if queue.complete(queue.snapshot()):
        complete(queue)
        return
    granted = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if len([x for x in granted.split(",") if x]) != 2:
        raise RuntimeError("This persistent Teacher requires exactly two allocated GPUs")
    for replica, devices in enumerate([granted]):
        backend = launch([sys.executable, "-m", "sglang.launch_server", "--model-path", str(MODEL),
                          "--served-model-name", "m", "--tp-size", "2", "--dtype", "bfloat16",
                          "--attention-backend", "triton", "--enable-fp32-lm-head",
                          "--mem-fraction-static", "0.85", "--host", "127.0.0.1", "--port", str(9080+replica),
                          "--engine-info-bootstrap-port", str(9780+replica)], f"teacher-backend-{replica}", devices)
        backend.port = 9080 + replica
    for backend in processes[:1]:
        wait_ready(backend.port, backend)
    frontends = []
    for replica in [0]:
        frontend = launch([sys.executable, str(MODEL / "knowline_server.py"), "--model", str(MODEL),
                           "--backend", "sglang", "--url", f"http://127.0.0.1:{9080+replica}",
                           "--served-model-name", "m", "--temperature", "1", "--workers", "16",
                           "--read-twice", "--twice-max-tokens", "3000", "--temperatures", str(MODEL / "temperatures.json"),
                           "--host", "127.0.0.1", "--port", str(18100+replica)], f"teacher-frontend-{replica}")
        frontends.append(frontend)
    for replica, frontend in enumerate(frontends):
        wait_ready(18100+replica, frontend, timeout=120)
    queue.start_annotation_clock()
    workers = [launch([sys.executable, "-u", str(CODE / "annotate_teacher.py"), "--replica", str(replica)], f"teacher-annotation-{replica}") for replica in [0]]
    print(json.dumps({"event": "teacher_annotation_started", "replicas": 1, "TP": 2,
                      "setup_seconds": time.monotonic()-started, "scheduling": "dynamic-state-local-chunks-v1",
                      "recovered_pairs": queue.snapshot()["recovered_pairs"]}), flush=True)
    while any(worker.poll() is None for worker in workers):
        if any(worker.poll() not in [None, 0] for worker in workers):
            raise RuntimeError("Teacher annotation worker failed; completed target rows are retained")
        if any(process.poll() is not None for process in processes[:2]):
            raise RuntimeError("Teacher backend/frontend exited during annotation")
        publish_progress(queue)
        time.sleep(10)
    if any(worker.returncode != 0 for worker in workers):
        raise RuntimeError("Teacher annotation worker exited unsuccessfully")
    complete(queue)


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    with (SOURCE_ROOT / "outputs/teacher/producer.lock").open("a+") as producer_lock, (CACHE / "stage.lock").open("a+") as stage_lock:
        fcntl.flock(producer_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(stage_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            main()
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for process in reversed(processes):
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            for handle in handles:
                handle.close()
