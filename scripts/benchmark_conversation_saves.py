"""Reproduce streaming persistence cost without a model or UI dependency.

Run from the repository root: python scripts/benchmark_conversation_saves.py
Each history has 20 turns/branch, 2 KiB/turn; report caller latency separately
from final flush latency. Temporary files never touch the real library.
"""

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chatlab import library  # noqa: E402
from chatlab.conversation import make_turn, new_forks, put_branch  # noqa: E402
from chatlab.library_writer import LibraryWriter  # noqa: E402


def measure(branches, frames, frame_interval=0):
    history = new_forks()
    for i in range(branches):
        put_branch(history, f"Chat {i}", [make_turn("user" if t % 2 == 0 else "assistant", "x" * 2048) for t in range(20)])
    results = {"branches": branches, "frames": frames, "frame_interval_seconds": frame_interval}
    with tempfile.TemporaryDirectory() as directory:
        for mode in ("synchronous", "coalesced"):
            path = Path(directory) / f"{mode}.json"
            library.write(history, path)
            results["file_bytes"] = path.stat().st_size
            writer = LibraryWriter() if mode == "coalesced" else None
            latencies = []
            count = 0
            write = library.write

            def counted(*args, **kwargs):
                nonlocal count
                count += 1
                return write(*args, **kwargs)

            from unittest.mock import patch
            try:
                with patch.object(library, "write", counted):
                    for frame in range(frames):
                        put_branch(history, "Main", [make_turn("assistant", "token " * (frame + 1))])
                        start = time.perf_counter()
                        if writer:
                            owned = {"active": "Main", "branches": {"Main": history["branches"]["Main"]}, "updated": {"Main": history["updated"]["Main"]}}
                            receipt = writer.submit("generation", owned, path)
                        else:
                            library.write(history, path, preserve_active=True)
                        latencies.append((time.perf_counter() - start) * 1000)
                        if frame_interval:
                            time.sleep(frame_interval)
                    start = time.perf_counter()
                    if writer:
                        assert writer.flush(receipt)
                    flush_ms = (time.perf_counter() - start) * 1000
            finally:
                if writer:
                    writer.close()
            results[mode] = {"caller_median_ms": round(statistics.median(latencies), 3), "caller_p95_ms": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 3), "caller_total_ms": round(sum(latencies), 3), "final_flush_ms": round(flush_ms, 3), "writes": count}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches", type=int, nargs="+", default=[10, 100, 500])
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--frame-interval", type=float, default=0, help="seconds between frames (0 for a burst; 0.05 for 20 frames/s)")
    args = parser.parse_args()
    if args.frames < 1 or min(args.branches) < 1 or args.frame_interval < 0:
        parser.error("frames and branches must be positive")
    print(json.dumps([measure(size, args.frames, args.frame_interval) for size in args.branches], indent=2))
