# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Timing instrumentation for analyzing contention between training and inference.

This module provides timing collection for:
- Scheduling time (lock contention analysis)
- Model execution time
- Batch counts

Metrics are collected separately for primary (inference) and secondary (training) paths.
"""

import csv
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class TimingStats:
    """Statistics for a single timing category."""
    count: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    total_tokens: int = 0  # For tracking tokens scheduled/executed
    
    def record(self, duration_ms: float, tokens: int = 0):
        self.count += 1
        self.total_time_ms += duration_ms
        self.min_time_ms = min(self.min_time_ms, duration_ms)
        self.max_time_ms = max(self.max_time_ms, duration_ms)
        self.total_tokens += tokens
    
    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / self.count if self.count > 0 else 0.0
    
    @property
    def avg_tokens_per_batch(self) -> float:
        return self.total_tokens / self.count if self.count > 0 else 0.0


@dataclass
class RequestStats:
    """Statistics for completed requests."""
    completed_requests: int = 0
    total_prompt_tokens: int = 0
    total_output_tokens: int = 0
    
    def record(self, prompt_tokens: int, output_tokens: int):
        self.completed_requests += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_output_tokens += output_tokens
    
    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_output_tokens
    
    @property
    def avg_prompt_tokens(self) -> float:
        return self.total_prompt_tokens / self.completed_requests if self.completed_requests > 0 else 0.0
    
    @property
    def avg_output_tokens(self) -> float:
        return self.total_output_tokens / self.completed_requests if self.completed_requests > 0 else 0.0


@dataclass
class TimingEvent:
    """A single timing event for CSV export."""
    timestamp: float  # Unix timestamp
    category: str  # 'schedule_primary', 'schedule_secondary', 'execute_primary', 'execute_secondary'
    duration_ms: float
    tokens: int
    batch_size: int  # Number of requests in the batch


class TimingCollector:
    """
    Collects timing metrics for scheduling and execution.
    
    Thread-safe collector that tracks:
    - schedule_primary: Time to schedule inference requests
    - schedule_secondary: Time to schedule training requests  
    - execute_primary: Time to execute inference batches
    - execute_secondary: Time to execute training batches
    """
    
    _instance: Optional["TimingCollector"] = None
    _lock = threading.Lock()
    
    def __init__(self, output_dir: Optional[str] = None):
        self._stats_lock = threading.Lock()
        
        # Aggregate statistics for timing
        self.schedule_primary = TimingStats()
        self.schedule_secondary = TimingStats()
        self.execute_primary = TimingStats()
        self.execute_secondary = TimingStats()
        
        # Aggregate statistics for completed requests
        self.requests_primary = RequestStats()
        self.requests_secondary = RequestStats()
        
        # Per-event logging for detailed analysis
        self._events: list[TimingEvent] = []
        self._max_events = 100000  # Limit memory usage
        
        # Output configuration
        self._output_dir = output_dir or os.environ.get(
            "VLLM_TIMING_OUTPUT_DIR", "/tmp/vllm_timing")
        self._enabled = os.environ.get("VLLM_TIMING_ENABLED", "0") == "1"
        
        if self._enabled:
            Path(self._output_dir).mkdir(parents=True, exist_ok=True)
            logger.info("Timing collection enabled, output dir: %s", self._output_dir)
    
    @classmethod
    def get_instance(cls) -> "TimingCollector":
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def enable(cls, output_dir: Optional[str] = None):
        """Enable timing collection."""
        instance = cls.get_instance()
        instance._enabled = True
        if output_dir:
            instance._output_dir = output_dir
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Timing collection enabled, output dir: %s", instance._output_dir)
    
    @classmethod
    def disable(cls):
        """Disable timing collection."""
        instance = cls.get_instance()
        instance._enabled = False
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    def record_schedule_primary(self, duration_ms: float, tokens: int, batch_size: int):
        """Record a primary (inference) scheduling event."""
        if not self._enabled:
            return
        with self._stats_lock:
            self.schedule_primary.record(duration_ms, tokens)
            self._add_event("schedule_primary", duration_ms, tokens, batch_size)
    
    def record_schedule_secondary(self, duration_ms: float, tokens: int, batch_size: int):
        """Record a secondary (training) scheduling event."""
        if not self._enabled:
            return
        with self._stats_lock:
            self.schedule_secondary.record(duration_ms, tokens)
            self._add_event("schedule_secondary", duration_ms, tokens, batch_size)
    
    def record_execute_primary(self, duration_ms: float, tokens: int, batch_size: int):
        """Record a primary (inference) execution event."""
        if not self._enabled:
            return
        with self._stats_lock:
            self.execute_primary.record(duration_ms, tokens)
            self._add_event("execute_primary", duration_ms, tokens, batch_size)
    
    def record_execute_secondary(self, duration_ms: float, tokens: int, batch_size: int):
        """Record a secondary (training) execution event."""
        if not self._enabled:
            return
        with self._stats_lock:
            self.execute_secondary.record(duration_ms, tokens)
            self._add_event("execute_secondary", duration_ms, tokens, batch_size)
    
    def record_request_completed_primary(self, prompt_tokens: int, output_tokens: int):
        """Record a completed primary (inference) request."""
        if not self._enabled:
            return
        with self._stats_lock:
            self.requests_primary.record(prompt_tokens, output_tokens)
    
    def record_request_completed_secondary(self, prompt_tokens: int, output_tokens: int):
        """Record a completed secondary (training) request."""
        if not self._enabled:
            return
        with self._stats_lock:
            self.requests_secondary.record(prompt_tokens, output_tokens)
    
    def _add_event(self, category: str, duration_ms: float, tokens: int, batch_size: int):
        """Add an event to the event log (caller must hold lock)."""
        if len(self._events) >= self._max_events:
            # Drop oldest events to prevent memory bloat
            self._events = self._events[self._max_events // 2:]
        
        self._events.append(TimingEvent(
            timestamp=time.time(),
            category=category,
            duration_ms=duration_ms,
            tokens=tokens,
            batch_size=batch_size,
        ))
    
    def get_summary(self) -> dict:
        """Get a summary of all timing statistics."""
        with self._stats_lock:
            return {
                "schedule_primary": {
                    "count": self.schedule_primary.count,
                    "total_time_ms": self.schedule_primary.total_time_ms,
                    "avg_time_ms": self.schedule_primary.avg_time_ms,
                    "min_time_ms": self.schedule_primary.min_time_ms if self.schedule_primary.count > 0 else 0,
                    "max_time_ms": self.schedule_primary.max_time_ms,
                    "total_tokens": self.schedule_primary.total_tokens,
                    "avg_tokens_per_batch": self.schedule_primary.avg_tokens_per_batch,
                },
                "schedule_secondary": {
                    "count": self.schedule_secondary.count,
                    "total_time_ms": self.schedule_secondary.total_time_ms,
                    "avg_time_ms": self.schedule_secondary.avg_time_ms,
                    "min_time_ms": self.schedule_secondary.min_time_ms if self.schedule_secondary.count > 0 else 0,
                    "max_time_ms": self.schedule_secondary.max_time_ms,
                    "total_tokens": self.schedule_secondary.total_tokens,
                    "avg_tokens_per_batch": self.schedule_secondary.avg_tokens_per_batch,
                },
                "execute_primary": {
                    "count": self.execute_primary.count,
                    "total_time_ms": self.execute_primary.total_time_ms,
                    "avg_time_ms": self.execute_primary.avg_time_ms,
                    "min_time_ms": self.execute_primary.min_time_ms if self.execute_primary.count > 0 else 0,
                    "max_time_ms": self.execute_primary.max_time_ms,
                    "total_tokens": self.execute_primary.total_tokens,
                    "avg_tokens_per_batch": self.execute_primary.avg_tokens_per_batch,
                },
                "execute_secondary": {
                    "count": self.execute_secondary.count,
                    "total_time_ms": self.execute_secondary.total_time_ms,
                    "avg_time_ms": self.execute_secondary.avg_time_ms,
                    "min_time_ms": self.execute_secondary.min_time_ms if self.execute_secondary.count > 0 else 0,
                    "max_time_ms": self.execute_secondary.max_time_ms,
                    "total_tokens": self.execute_secondary.total_tokens,
                    "avg_tokens_per_batch": self.execute_secondary.avg_tokens_per_batch,
                },
            }
    
    def get_request_summary(self) -> dict:
        """Get a summary of completed request statistics."""
        with self._stats_lock:
            return {
                "primary_inference": {
                    "completed_requests": self.requests_primary.completed_requests,
                    "total_prompt_tokens": self.requests_primary.total_prompt_tokens,
                    "total_output_tokens": self.requests_primary.total_output_tokens,
                    "total_tokens": self.requests_primary.total_tokens,
                    "avg_prompt_tokens": self.requests_primary.avg_prompt_tokens,
                    "avg_output_tokens": self.requests_primary.avg_output_tokens,
                },
                "secondary_training": {
                    "completed_requests": self.requests_secondary.completed_requests,
                    "total_prompt_tokens": self.requests_secondary.total_prompt_tokens,
                    "total_output_tokens": self.requests_secondary.total_output_tokens,
                    "total_tokens": self.requests_secondary.total_tokens,
                    "avg_prompt_tokens": self.requests_secondary.avg_prompt_tokens,
                    "avg_output_tokens": self.requests_secondary.avg_output_tokens,
                },
            }
    
    def export_summary_csv(self, filename: Optional[str] = None) -> str:
        """Export summary statistics to CSV."""
        if filename is None:
            filename = os.path.join(self._output_dir, "timing_summary.csv")
        
        summary = self.get_summary()
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "category", "count", "total_time_ms", "avg_time_ms", 
                "min_time_ms", "max_time_ms", "total_tokens", "avg_tokens_per_batch"
            ])
            for category, stats in summary.items():
                writer.writerow([
                    category,
                    stats["count"],
                    f"{stats['total_time_ms']:.3f}",
                    f"{stats['avg_time_ms']:.3f}",
                    f"{stats['min_time_ms']:.3f}",
                    f"{stats['max_time_ms']:.3f}",
                    stats["total_tokens"],
                    f"{stats['avg_tokens_per_batch']:.2f}",
                ])
        
        logger.info("Exported timing summary to %s", filename)
        return filename
    
    def export_events_csv(self, filename: Optional[str] = None) -> str:
        """Export detailed events to CSV for time-series analysis."""
        if filename is None:
            filename = os.path.join(self._output_dir, "timing_events.csv")
        
        with self._stats_lock:
            events = list(self._events)
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "category", "duration_ms", "tokens", "batch_size"])
            for event in events:
                writer.writerow([
                    f"{event.timestamp:.6f}",
                    event.category,
                    f"{event.duration_ms:.3f}",
                    event.tokens,
                    event.batch_size,
                ])
        
        logger.info("Exported %d timing events to %s", len(events), filename)
        return filename
    
    def export_requests_csv(self, filename: Optional[str] = None) -> str:
        """Export request completion statistics to CSV."""
        if filename is None:
            filename = os.path.join(self._output_dir, "request_stats.csv")
        
        summary = self.get_request_summary()
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "category", "completed_requests", "total_prompt_tokens", 
                "total_output_tokens", "total_tokens", "avg_prompt_tokens", "avg_output_tokens"
            ])
            for category, stats in summary.items():
                writer.writerow([
                    category,
                    stats["completed_requests"],
                    stats["total_prompt_tokens"],
                    stats["total_output_tokens"],
                    stats["total_tokens"],
                    f"{stats['avg_prompt_tokens']:.2f}",
                    f"{stats['avg_output_tokens']:.2f}",
                ])
        
        logger.info("Exported request stats to %s", filename)
        return filename
    
    def export_all(self, prefix: Optional[str] = None) -> tuple[str, str, str]:
        """Export summary, events, and request stats CSVs."""
        if prefix:
            summary_file = os.path.join(self._output_dir, f"{prefix}_timing_summary.csv")
            events_file = os.path.join(self._output_dir, f"{prefix}_timing_events.csv")
            requests_file = os.path.join(self._output_dir, f"{prefix}_request_stats.csv")
        else:
            summary_file = None
            events_file = None
            requests_file = None
        
        return (
            self.export_summary_csv(summary_file),
            self.export_events_csv(events_file),
            self.export_requests_csv(requests_file),
        )
    
    def reset(self):
        """Reset all statistics."""
        with self._stats_lock:
            self.schedule_primary = TimingStats()
            self.schedule_secondary = TimingStats()
            self.execute_primary = TimingStats()
            self.execute_secondary = TimingStats()
            self.requests_primary = RequestStats()
            self.requests_secondary = RequestStats()
            self._events.clear()
    
    def print_summary(self):
        """Print a human-readable summary to the logger."""
        summary = self.get_summary()
        request_summary = self.get_request_summary()
        
        logger.info("=" * 60)
        logger.info("TIMING SUMMARY")
        logger.info("=" * 60)
        
        for category, stats in summary.items():
            if stats["count"] > 0:
                logger.info(
                    "%s: count=%d, total=%.1fms, avg=%.3fms, min=%.3fms, max=%.3fms, "
                    "tokens=%d, avg_tokens=%.1f",
                    category,
                    stats["count"],
                    stats["total_time_ms"],
                    stats["avg_time_ms"],
                    stats["min_time_ms"],
                    stats["max_time_ms"],
                    stats["total_tokens"],
                    stats["avg_tokens_per_batch"],
                )
        
        logger.info("-" * 60)
        logger.info("REQUEST COMPLETION SUMMARY")
        logger.info("-" * 60)
        
        for category, stats in request_summary.items():
            if stats["completed_requests"] > 0:
                logger.info(
                    "%s: completed=%d, prompt_tokens=%d (avg=%.1f), "
                    "output_tokens=%d (avg=%.1f), total_tokens=%d",
                    category,
                    stats["completed_requests"],
                    stats["total_prompt_tokens"],
                    stats["avg_prompt_tokens"],
                    stats["total_output_tokens"],
                    stats["avg_output_tokens"],
                    stats["total_tokens"],
                )
        
        logger.info("=" * 60)


# Convenience functions for timing
class TimingContext:
    """Context manager for timing a block of code."""
    
    def __init__(self, record_fn, tokens: int = 0, batch_size: int = 0):
        self.record_fn = record_fn
        self.tokens = tokens
        self.batch_size = batch_size
        self.start_time = 0.0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        self.record_fn(duration_ms, self.tokens, self.batch_size)
        return False


def time_schedule_primary(tokens: int = 0, batch_size: int = 0) -> TimingContext:
    """Context manager for timing primary scheduling."""
    collector = TimingCollector.get_instance()
    return TimingContext(collector.record_schedule_primary, tokens, batch_size)


def time_schedule_secondary(tokens: int = 0, batch_size: int = 0) -> TimingContext:
    """Context manager for timing secondary scheduling."""
    collector = TimingCollector.get_instance()
    return TimingContext(collector.record_schedule_secondary, tokens, batch_size)


def time_execute_primary(tokens: int = 0, batch_size: int = 0) -> TimingContext:
    """Context manager for timing primary execution."""
    collector = TimingCollector.get_instance()
    return TimingContext(collector.record_execute_primary, tokens, batch_size)


def time_execute_secondary(tokens: int = 0, batch_size: int = 0) -> TimingContext:
    """Context manager for timing secondary execution."""
    collector = TimingCollector.get_instance()
    return TimingContext(collector.record_execute_secondary, tokens, batch_size)

