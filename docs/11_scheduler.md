# Phase 3: Continuous Batching Scheduler

## Table of Contents

1. [Understanding Continuous Batching](#understanding-continuous-batching)
2. [Scheduler Architecture](#scheduler-architecture)
3. [Request Lifecycle](#request-lifecycle)
4. [Implementation](#implementation)
5. [Scheduling Policies](#scheduling-policies)
6. [Integration with KV Cache](#integration-with-kv-cache)
7. [Testing and Verification](#testing-and-verification)

---

## Understanding Continuous Batching

**Continuous batching** dynamically adds and removes sequences from execution batches, maximizing GPU utilization.

```
                    Static vs Continuous Batching

STATIC BATCHING (Traditional):
┌────────────────────────────────────────────────────────┐
│ Wait for batch to fill → Process all → Wait again     │
│                                                        │
│ Time ──────────────────────────────────────────────▶  │
│                                                        │
│ [───Waiting───][█████ Batch 1 █████][───Waiting───]   │
│ [                        ][█████ Batch 2 █████]       │
│                                                        │
│ Problem: Sequences finish at different times!         │
│ - Short sequences: wait for long ones                  │
│ - Long sequences: block new requests                  │
│ - GPU underutilized during waiting                    │
└────────────────────────────────────────────────────────┘

CONTINUOUS BATCHING (vLLM/SGLang):
┌────────────────────────────────────────────────────────┐
│ Process whatever is ready, add/remove dynamically     │
│                                                        │
│ Time ──────────────────────────────────────────────▶  │
│                                                        │
│ Req A: [████████████████████]                         │
│ Req B:   [████████]           ← Finished early        │
│ Req C:     [██████████████████████████]               │
│ Req D:       [████]   ← New request added             │
│ Req E:         [████████████]                         │
│                                                        │
│ GPU always has work! Maximum throughput! 🚀          │
└────────────────────────────────────────────────────────┘
```

### Why It Matters

```
Throughput comparison (7B model, 4096 context):

Static Batching:
- Batch size: 8
- Avg latency: 500ms
- Throughput: 16 req/s
- GPU util: 40%

Continuous Batching:
- Dynamic batch: 8-64
- Avg latency: 200ms
- Throughput: 80 req/s  ← 5x improvement!
- GPU util: 95%
```

---

## Scheduler Architecture

```
                    Scheduler Components

┌─────────────────────────────────────────────────────────┐
│                      Scheduler                          │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Waiting Queue                       │   │
│  │                                                  │   │
│  │  [Req 5] [Req 6] [Req 7] ... (pending requests) │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         ▼ schedule()                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Running Queue                       │   │
│  │                                                  │   │
│  │  [Req 1] [Req 2] [Req 3] [Req 4] (executing)   │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Finished Queue                      │   │
│  │                                                  │   │
│  │  Completed requests ready for response          │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Preempted Queue                     │   │
│  │                                                  │   │
│  │  Requests paused due to memory pressure         │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Request Lifecycle

```
                    Request State Machine

                        ┌─────────┐
                        │ PENDING │
                        └────┬────┘
                             │ schedule()
                             ▼
                    ┌────────────────┐
                    │   PREFILLING   │  ← Processing prompt
                    └───────┬────────┘
                            │ prompt done
                            ▼
                    ┌────────────────┐
             ┌──────│   DECODING     │◀─────┐
             │      └───────┬────────┘      │
             │              │               │ continue
    preempt  │              │ token done    │
             │              ▼               │
             │      ┌────────────────┐      │
             │      │  Check Done?   │──────┘
             │      └───────┬────────┘
             │              │ yes (EOS or max_len)
             │              ▼
             │      ┌────────────────┐
             │      │   FINISHED     │
             │      └────────────────┘
             │
             ▼
     ┌────────────────┐
     │   PREEMPTED   │  ← Waiting to resume
     └───────┬────────┘
             │ memory available
             ▼
     ┌────────────────┐
     │   DECODING     │  ← Resume generation
     └────────────────┘
```

---

## Implementation

Create file: `mini_vllm/python/mini_vllm/scheduler.py`

```python
"""
Continuous Batching Scheduler
==============================

Implements the core scheduling logic for continuous batching.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum, auto
from collections import deque
import time


class SequenceStatus(Enum):
    """Status of a sequence in the scheduler"""
    PENDING = auto()      # Waiting to start
    PREFILLING = auto()   # Processing prompt
    DECODING = auto()     # Generating tokens
    PREEMPTED = auto()    # Paused due to memory
    FINISHED = auto()     # Complete
    ABORTED = auto()      # Cancelled


@dataclass
class SamplingParams:
    """Parameters for token sampling"""
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 256
    stop_sequences: List[str] = field(default_factory=list)


@dataclass
class SequenceData:
    """Data for a single sequence"""
    seq_id: int
    prompt_tokens: List[int]
    sampling_params: SamplingParams

    # State
    status: SequenceStatus = SequenceStatus.PENDING
    output_tokens: List[int] = field(default_factory=list)

    # Timing
    arrival_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    finish_time: Optional[float] = None

    # KV cache
    block_ids: List[int] = field(default_factory=list)
    num_computed_tokens: int = 0

    @property
    def num_tokens(self) -> int:
        """Total tokens (prompt + output)"""
        return len(self.prompt_tokens) + len(self.output_tokens)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_tokens)

    @property
    def is_finished(self) -> bool:
        return self.status in (SequenceStatus.FINISHED, SequenceStatus.ABORTED)

    @property
    def waiting_time(self) -> float:
        if self.start_time is None:
            return time.time() - self.arrival_time
        return self.start_time - self.arrival_time


@dataclass
class SchedulerOutput:
    """Output from a scheduling step"""
    # Sequences to execute
    prefill_seqs: List[SequenceData] = field(default_factory=list)
    decode_seqs: List[SequenceData] = field(default_factory=list)

    # Sequences that finished
    finished_seqs: List[SequenceData] = field(default_factory=list)

    # Block operations
    blocks_to_swap_in: Dict[int, int] = field(default_factory=dict)
    blocks_to_swap_out: Dict[int, int] = field(default_factory=dict)
    blocks_to_copy: Dict[int, int] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return len(self.prefill_seqs) == 0 and len(self.decode_seqs) == 0


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler"""
    max_num_seqs: int = 256          # Max concurrent sequences
    max_num_tokens: int = 8192       # Max tokens per step
    max_prefill_tokens: int = 4096   # Max tokens for one prefill

    # Batching
    max_batch_size: int = 64
    chunk_prefill: bool = True       # Split long prefills
    chunk_size: int = 512

    # Priority
    priority_preempted: bool = True  # Resume preempted first


class Scheduler:
    """
    Continuous Batching Scheduler

    Manages sequence queues and decides what to execute each step.
    """

    def __init__(
        self,
        config: SchedulerConfig,
        block_manager  # Type: BlockManager from kv_cache.py
    ):
        self.config = config
        self.block_manager = block_manager

        # Queues
        self.waiting: deque[SequenceData] = deque()
        self.running: List[SequenceData] = []
        self.preempted: deque[SequenceData] = deque()
        self.finished: List[SequenceData] = []

        # ID tracking
        self._next_seq_id = 0
        self._seq_map: Dict[int, SequenceData] = {}

        # Stats
        self.num_scheduled_tokens = 0
        self.num_scheduled_seqs = 0

    def add_request(
        self,
        prompt_tokens: List[int],
        sampling_params: Optional[SamplingParams] = None
    ) -> int:
        """
        Add a new request to the waiting queue.

        Returns sequence ID for tracking.
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        seq_id = self._next_seq_id
        self._next_seq_id += 1

        seq = SequenceData(
            seq_id=seq_id,
            prompt_tokens=prompt_tokens,
            sampling_params=sampling_params,
        )

        self.waiting.append(seq)
        self._seq_map[seq_id] = seq

        return seq_id

    def abort_request(self, seq_id: int):
        """Abort a request"""
        if seq_id not in self._seq_map:
            return

        seq = self._seq_map[seq_id]
        seq.status = SequenceStatus.ABORTED

        # Free KV cache
        if seq.block_ids:
            for block_id in seq.block_ids:
                self.block_manager.free(block_id)
            seq.block_ids = []

        # Remove from queues
        if seq in self.running:
            self.running.remove(seq)
        elif seq in self.waiting:
            self.waiting.remove(seq)
        elif seq in self.preempted:
            self.preempted.remove(seq)

        self.finished.append(seq)

    def schedule(self) -> SchedulerOutput:
        """
        Main scheduling function.

        Called each step to determine what to execute.
        """
        output = SchedulerOutput()

        # 1. Process finished sequences
        self._process_finished(output)

        # 2. Try to resume preempted sequences first
        if self.config.priority_preempted:
            self._schedule_preempted(output)

        # 3. Continue running sequences (decode step)
        self._schedule_running(output)

        # 4. Start new sequences if we have capacity
        self._schedule_waiting(output)

        # Update stats
        self.num_scheduled_seqs += len(output.prefill_seqs) + len(output.decode_seqs)
        for seq in output.prefill_seqs:
            self.num_scheduled_tokens += seq.num_prompt_tokens
        self.num_scheduled_tokens += len(output.decode_seqs)  # 1 token each

        return output

    def _process_finished(self, output: SchedulerOutput):
        """Move finished sequences out of running"""
        still_running = []

        for seq in self.running:
            if seq.is_finished:
                output.finished_seqs.append(seq)
                # Free KV cache blocks
                for block_id in seq.block_ids:
                    self.block_manager.free(block_id)
                seq.block_ids = []
            else:
                still_running.append(seq)

        self.running = still_running

    def _schedule_preempted(self, output: SchedulerOutput):
        """Try to resume preempted sequences"""
        while self.preempted:
            if len(self.running) >= self.config.max_batch_size:
                break

            seq = self.preempted[0]

            # Check if we can allocate blocks
            blocks_needed = self._blocks_needed(seq.num_tokens)
            if not self.block_manager.can_allocate(blocks_needed):
                break

            # Resume
            self.preempted.popleft()
            self._allocate_blocks(seq, blocks_needed)
            seq.status = SequenceStatus.DECODING

            self.running.append(seq)
            output.decode_seqs.append(seq)

    def _schedule_running(self, output: SchedulerOutput):
        """Schedule decode step for running sequences"""
        tokens_budget = self.config.max_num_tokens

        for seq in self.running:
            if seq.status == SequenceStatus.DECODING:
                # Check if we need to allocate new blocks
                blocks_needed = self._blocks_needed(seq.num_tokens + 1)
                current_blocks = len(seq.block_ids)

                if blocks_needed > current_blocks:
                    if self.block_manager.can_allocate(1):
                        new_block = self.block_manager.allocate()
                        seq.block_ids.append(new_block)
                    else:
                        # Need to preempt
                        self._preempt_sequence(seq)
                        continue

                output.decode_seqs.append(seq)
                tokens_budget -= 1

    def _schedule_waiting(self, output: SchedulerOutput):
        """Start new sequences from waiting queue"""
        tokens_budget = self.config.max_prefill_tokens

        while self.waiting:
            if len(self.running) >= self.config.max_batch_size:
                break

            seq = self.waiting[0]

            # Chunked prefill: only process chunk_size tokens at a time
            if self.config.chunk_prefill:
                tokens_to_process = min(
                    seq.num_prompt_tokens - seq.num_computed_tokens,
                    self.config.chunk_size
                )
            else:
                tokens_to_process = seq.num_prompt_tokens

            if tokens_to_process > tokens_budget:
                break

            # Check block capacity
            blocks_needed = self._blocks_needed(seq.num_prompt_tokens)
            if not self.block_manager.can_allocate(blocks_needed):
                # Try preempting running sequences
                if not self._try_preempt_for_new(blocks_needed):
                    break

            # Start this sequence
            self.waiting.popleft()
            self._allocate_blocks(seq, blocks_needed)

            seq.status = SequenceStatus.PREFILLING
            seq.start_time = time.time()

            self.running.append(seq)
            output.prefill_seqs.append(seq)

            tokens_budget -= tokens_to_process

    def _blocks_needed(self, num_tokens: int) -> int:
        """Calculate blocks needed for given token count"""
        block_size = self.block_manager.config.block_size
        return (num_tokens + block_size - 1) // block_size

    def _allocate_blocks(self, seq: SequenceData, num_blocks: int):
        """Allocate KV cache blocks for sequence"""
        current = len(seq.block_ids)
        needed = num_blocks - current

        if needed > 0:
            new_blocks = self.block_manager.allocate_blocks(needed)
            seq.block_ids.extend(new_blocks)

    def _preempt_sequence(self, seq: SequenceData):
        """Preempt a sequence (pause execution)"""
        seq.status = SequenceStatus.PREEMPTED

        # Free blocks (will need to recompute KV)
        for block_id in seq.block_ids:
            self.block_manager.free(block_id)
        seq.block_ids = []

        self.running.remove(seq)
        self.preempted.append(seq)

    def _try_preempt_for_new(self, blocks_needed: int) -> bool:
        """Try to preempt running sequences to make room"""
        # Find sequence with lowest priority (FIFO: longest running)
        if not self.running:
            return False

        # Simple: preempt last-started sequence
        for seq in reversed(self.running):
            if seq.status == SequenceStatus.DECODING:
                blocks_freed = len(seq.block_ids)
                self._preempt_sequence(seq)

                if self.block_manager.can_allocate(blocks_needed):
                    return True

        return False

    def update_sequence(self, seq_id: int, new_token: int) -> bool:
        """
        Update sequence with newly generated token.

        Returns True if sequence should continue, False if finished.
        """
        if seq_id not in self._seq_map:
            return False

        seq = self._seq_map[seq_id]
        seq.output_tokens.append(new_token)

        # Check termination conditions
        sampling = seq.sampling_params

        # Max tokens
        if len(seq.output_tokens) >= sampling.max_tokens:
            seq.status = SequenceStatus.FINISHED
            seq.finish_time = time.time()
            return False

        # EOS token (assuming EOS = 2, adjust for your tokenizer)
        if new_token == 2:
            seq.status = SequenceStatus.FINISHED
            seq.finish_time = time.time()
            return False

        # After prefill, switch to decode
        if seq.status == SequenceStatus.PREFILLING:
            if seq.num_computed_tokens >= seq.num_prompt_tokens:
                seq.status = SequenceStatus.DECODING

        return True

    def get_sequence(self, seq_id: int) -> Optional[SequenceData]:
        """Get sequence data by ID"""
        return self._seq_map.get(seq_id)

    def get_num_running(self) -> int:
        return len(self.running)

    def get_num_waiting(self) -> int:
        return len(self.waiting)

    def get_num_preempted(self) -> int:
        return len(self.preempted)

    def get_stats(self) -> Dict:
        """Get scheduler statistics"""
        return {
            "num_waiting": self.get_num_waiting(),
            "num_running": self.get_num_running(),
            "num_preempted": self.get_num_preempted(),
            "num_finished": len(self.finished),
            "total_scheduled_tokens": self.num_scheduled_tokens,
            "total_scheduled_seqs": self.num_scheduled_seqs,
        }


class BatchState:
    """
    Represents the current batch state for model execution.

    Combines prefill and decode sequences into a single batch.
    """

    def __init__(self, output: SchedulerOutput, block_size: int):
        self.prefill_seqs = output.prefill_seqs
        self.decode_seqs = output.decode_seqs
        self.block_size = block_size

        # Build batch tensors
        self._build_batch()

    def _build_batch(self):
        """Build batch metadata for model execution"""
        self.seq_ids: List[int] = []
        self.positions: List[int] = []
        self.block_tables: List[List[int]] = []
        self.context_lens: List[int] = []

        # Prefill sequences
        for seq in self.prefill_seqs:
            self.seq_ids.append(seq.seq_id)
            # Positions for all prompt tokens
            start_pos = seq.num_computed_tokens
            end_pos = seq.num_prompt_tokens
            self.positions.extend(range(start_pos, end_pos))
            self.block_tables.append(seq.block_ids.copy())
            self.context_lens.append(seq.num_prompt_tokens)

        # Decode sequences
        for seq in self.decode_seqs:
            self.seq_ids.append(seq.seq_id)
            # Position is current length
            self.positions.append(seq.num_tokens - 1)
            self.block_tables.append(seq.block_ids.copy())
            self.context_lens.append(seq.num_tokens)

    @property
    def num_prefill_tokens(self) -> int:
        return sum(s.num_prompt_tokens - s.num_computed_tokens
                   for s in self.prefill_seqs)

    @property
    def num_decode_tokens(self) -> int:
        return len(self.decode_seqs)

    @property
    def is_empty(self) -> bool:
        return len(self.prefill_seqs) == 0 and len(self.decode_seqs) == 0
```

---

## Scheduling Policies

### FCFS (First-Come-First-Served)

```python
# Default policy: process in arrival order
# Already implemented in Scheduler.schedule()
```

### Shortest-Job-First (SJF)

```python
def schedule_sjf(self, output: SchedulerOutput):
    """Schedule shorter prompts first"""
    # Sort waiting queue by prompt length
    sorted_waiting = sorted(self.waiting, key=lambda s: s.num_prompt_tokens)

    for seq in sorted_waiting:
        if len(self.running) >= self.config.max_batch_size:
            break
        # ... rest of scheduling logic
```

### Priority-based

```python
@dataclass
class SequenceData:
    # ... existing fields
    priority: int = 0  # Higher = more important

def schedule_priority(self, output: SchedulerOutput):
    """Schedule by priority, then by arrival time"""
    sorted_waiting = sorted(
        self.waiting,
        key=lambda s: (-s.priority, s.arrival_time)
    )
```

---

## Integration with KV Cache

```python
class SchedulerWithRadix:
    """
    Scheduler integrated with radix attention for prefix caching.
    """

    def __init__(self, config, kv_cache_manager, radix_tree):
        self.config = config
        self.kv_cache = kv_cache_manager
        self.radix = radix_tree
        # ... queues as before

    def schedule_with_prefix(self, output: SchedulerOutput):
        """Schedule with prefix matching"""
        for seq in self.waiting:
            # Check for cached prefix
            match = self.radix.match(seq.prompt_tokens)

            if match.matched_tokens > 0:
                # Reuse cached blocks
                seq.block_ids = match.matched_blocks.copy()
                seq.num_computed_tokens = match.matched_tokens

                # Add references to shared blocks
                for block_id in match.matched_blocks:
                    self.kv_cache.add_ref(block_id)

            # Only allocate for non-cached part
            remaining_tokens = seq.num_prompt_tokens - seq.num_computed_tokens
            blocks_needed = (remaining_tokens + self.block_size - 1) // self.block_size

            if self.kv_cache.can_allocate(blocks_needed):
                new_blocks = self.kv_cache.allocate_blocks(blocks_needed)
                seq.block_ids.extend(new_blocks)

                # Insert into radix tree
                self.radix.insert(seq.prompt_tokens, seq.block_ids)
```

---

## Testing and Verification

Create file: `mini_vllm/tests/python/test_scheduler.py`

```python
"""
Test Scheduler Implementation
"""

import pytest
from unittest.mock import MagicMock
from mini_vllm.scheduler import (
    Scheduler, SchedulerConfig, SchedulerOutput,
    SequenceData, SequenceStatus, SamplingParams
)


class MockBlockManager:
    """Mock block manager for testing"""
    def __init__(self, num_blocks=100, block_size=16):
        self.config = MagicMock()
        self.config.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.allocated = {}

    def can_allocate(self, n):
        return len(self.free_blocks) >= n

    def allocate(self):
        if not self.free_blocks:
            raise RuntimeError("No blocks")
        block = self.free_blocks.pop()
        self.allocated[block] = 1
        return block

    def allocate_blocks(self, n):
        return [self.allocate() for _ in range(n)]

    def free(self, block_id):
        if block_id in self.allocated:
            del self.allocated[block_id]
            self.free_blocks.append(block_id)


class TestScheduler:
    @pytest.fixture
    def scheduler(self):
        config = SchedulerConfig(
            max_num_seqs=256,
            max_batch_size=8,
            max_prefill_tokens=1024,
        )
        block_manager = MockBlockManager()
        return Scheduler(config, block_manager)

    def test_add_request(self, scheduler):
        seq_id = scheduler.add_request([1, 2, 3, 4, 5])
        assert seq_id == 0
        assert len(scheduler.waiting) == 1

    def test_schedule_prefill(self, scheduler):
        scheduler.add_request([1, 2, 3, 4, 5])
        scheduler.add_request([10, 20, 30])

        output = scheduler.schedule()

        assert len(output.prefill_seqs) == 2
        assert len(scheduler.running) == 2
        assert len(scheduler.waiting) == 0

    def test_schedule_decode(self, scheduler):
        seq_id = scheduler.add_request([1, 2, 3])

        # First schedule: prefill
        output1 = scheduler.schedule()
        assert len(output1.prefill_seqs) == 1

        # Simulate prefill complete
        seq = scheduler.get_sequence(seq_id)
        seq.num_computed_tokens = 3
        seq.status = SequenceStatus.DECODING

        # Second schedule: decode
        output2 = scheduler.schedule()
        assert len(output2.decode_seqs) == 1

    def test_max_batch_size(self, scheduler):
        # Add more than max_batch_size
        for i in range(10):
            scheduler.add_request([i])

        output = scheduler.schedule()

        # Should only schedule max_batch_size
        assert len(output.prefill_seqs) == scheduler.config.max_batch_size
        assert len(scheduler.waiting) == 2

    def test_finish_sequence(self, scheduler):
        seq_id = scheduler.add_request([1, 2, 3])

        # Schedule and mark finished
        scheduler.schedule()
        seq = scheduler.get_sequence(seq_id)
        seq.status = SequenceStatus.FINISHED

        output = scheduler.schedule()

        assert len(output.finished_seqs) == 1
        assert len(scheduler.running) == 0

    def test_abort_request(self, scheduler):
        seq_id = scheduler.add_request([1, 2, 3])

        scheduler.abort_request(seq_id)

        seq = scheduler.get_sequence(seq_id)
        assert seq.status == SequenceStatus.ABORTED

    def test_update_sequence(self, scheduler):
        seq_id = scheduler.add_request(
            [1, 2, 3],
            SamplingParams(max_tokens=5)
        )
        scheduler.schedule()

        # Add tokens
        assert scheduler.update_sequence(seq_id, 10)
        assert scheduler.update_sequence(seq_id, 20)

        seq = scheduler.get_sequence(seq_id)
        assert seq.output_tokens == [10, 20]

    def test_max_tokens_finish(self, scheduler):
        seq_id = scheduler.add_request(
            [1, 2, 3],
            SamplingParams(max_tokens=2)
        )
        scheduler.schedule()
        seq = scheduler.get_sequence(seq_id)
        seq.status = SequenceStatus.DECODING

        scheduler.update_sequence(seq_id, 10)
        result = scheduler.update_sequence(seq_id, 20)

        assert not result  # Should indicate finished
        assert seq.status == SequenceStatus.FINISHED

    def test_stats(self, scheduler):
        scheduler.add_request([1, 2, 3])
        scheduler.add_request([4, 5, 6])
        scheduler.schedule()

        stats = scheduler.get_stats()
        assert stats["num_running"] == 2
        assert stats["num_waiting"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## Summary

You've implemented a continuous batching scheduler with:

| Feature              | Implementation                        |
| -------------------- | ------------------------------------- |
| **Queue management** | Waiting, running, preempted, finished |
| **Batch building**   | Combine prefill and decode            |
| **Preemption**       | Handle memory pressure                |
| **Chunked prefill**  | Process long prompts in pieces        |

### Key Concepts

1. **Continuous batching** - Dynamic batch composition
2. **Sequence lifecycle** - PENDING → PREFILLING → DECODING → FINISHED
3. **Preemption** - Pause sequences when memory tight
4. **Integration** - Works with KV cache and radix tree

---

## What's Next

Next, we'll implement the **Model Runner** that executes the model on batches.

Continue to: [12_model_runner.md](./12_model_runner.md)
