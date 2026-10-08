# Conversation save scheduling and measurements

Streaming persistence runs on one dedicated `chatlab-library-writer` thread.
The generation worker snapshots its owned conversation and enqueues it instead
of reading, merging, serializing, and replacing the whole library on each frame.
Token forks also carry their source branch's inherited sampling state.
Each producer has one pending snapshot and at most one in flight. Later frames
replace the pending snapshot without postponing the deadline: the writer starts
saving 500 ms after the first pending update, even during continuous streaming.

Completion, inference failure, and cancellation finalize the partial reply and
wait for the final save outside the job lock. Polling and Stop can still acquire
that lock while storage is busy. The job remains running until its final save
finishes, so subsequent edits cannot race its completion. Desktop shutdown,
checkout server shutdown, remote standard-input closure, and normal Python exit
drain all accepted snapshots and reject subsequent streaming saves. Shutdown
saves the latest published frame without waiting for potentially stalled model
inference. A write failure is logged and releases the flush barrier; a subsequent
frame can attempt another save.

Navigation, explicit edits, and atomic branch-name reservations retain their
synchronous saves. Queued frames still use `library.write` and its process-wide
read/merge/replace lock. Transcript timestamps, independent sampling/archive
timestamps, deletion tombstones, and the current selection on disk remain the
merge authority. Partial saves retain the latest pane order on disk and append
new branches in snapshot order; they cannot reorder unrelated conversations.
Multiple processes writing the same file remain unsupported.

## Accepted crash-loss window

With healthy storage and a scheduled writer, an abrupt process crash can lose
**500 ms plus write and scheduling time** of streamed output. Continuous updates
do not postpone saves. Completion and cancellation wait for their final save;
orderly shutdown drains every accepted frame. Disk failures, storage stalls,
and a backed-up writer can extend the window. Atomic replacement retains a
complete previous file if a process dies during a write. Files are not fsynced,
so this is not a power-loss durability guarantee.

## Reproduce the benchmark

From the repository root:

```sh
python3 scripts/benchmark_conversation_saves.py
python3 scripts/benchmark_conversation_saves.py --frame-interval 0.05
```

The benchmark uses temporary files, 20 turns per conversation, and 2 KiB of text
per turn. It publishes 30 updates to a separate generating conversation. The
synchronous baseline saves the full pane using the unchanged `library.write`;
the new path enqueues only the owned branch. Caller timing includes snapshot and
queue work, or the full synchronous write. Final flush timing is reported
separately. Frame construction and `put_branch` run outside that timing in both
cases. No model is loaded.

Measured on macOS 27.0.1, arm64, Python 3.14.4, October 8, 2026:

| History | Library size | Synchronous median / p95 | Queued median / p95 | Writes before → after |
| --- | --- | --- | --- | --- |
| 10 conversations | 0.43 MB | 2.35 / 2.53 ms | 0.046 / 0.065 ms | 30 → 3 |
| 100 conversations | 4.31 MB | 20.22 / 23.32 ms | 0.044 / 0.052 ms | 30 → 3 |
| 500 conversations | 21.54 MB | 103.71 / 113.55 ms | 0.044 / 0.055 ms | 30 → 3 |

These are paced results, with 50 ms between updates. The final queued flush took
72 ms at 500 conversations. In an unpaced burst, 30 updates coalesced into one
write; at 500 conversations median caller time fell from 98.41 ms to 0.006 ms,
with a 113 ms final flush. Both runs used cached local storage and synthetic
transcripts. They demonstrate lower persistence latency and fewer rewrites;
they do not measure end-to-end UI responsiveness, model throughput, or eliminate
CPU contention from Python serialization on the writer thread.
