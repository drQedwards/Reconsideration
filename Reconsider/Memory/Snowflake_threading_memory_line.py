# We'll create the requested Python module in /mnt/data and print a short summary of what's inside.
# The module includes:
# - Snowflake ID generator
# - InMemory and Redis-like stubs for MemoryStore
# - Comment threading data structures (posts, comments, threads)
# - ConsensusGraph stub with vector-similarity placeholder
# - Reconsideration engine with tracing
# - A simple CLI demo (`--demo`) that simulates users, posts, and a reconsideration pass
#
# NOTE: This code is self-contained and runnable in a standard Python 3.10+ environment
# without external dependencies. Redis/vector DB integrations are structured as optional stubs.

from textwrap import dedent
from pathlib import Path

code = dedent('''
#!/usr/bin/env python3
"""
snowflake_reconsideraiton_comment_threading_trace.py

A reference implementation of a memory reconsideration pipeline with:
- 64-bit Snowflake IDs
- Comment threading (posts, comments, reply chains)
- Consensus-driven confidence updates
- Temporal decay, contradiction penalties
- Per-step tracing via logging
- In-memory store (with Redis-like interface stubs)
- Demo CLI to simulate users and run a reconsideration pass

Author: Dr Q
License: MIT
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Iterable
import time
import logging
import random
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------------------------------------
# Logging / Tracing
# ------------------------------------------------------------

LOGGER = logging.getLogger("reconsider")
HANDLER = logging.StreamHandler()
FORMAT  = "%(asctime)s | %(levelname)-7s | %(message)s"
HANDLER.setFormatter(logging.Formatter(FORMAT))
LOGGER.addHandler(HANDLER)
LOGGER.setLevel(logging.INFO)


# ------------------------------------------------------------
# Snowflake (Twitter-like, 64-bit): time + worker + sequence
# ------------------------------------------------------------

class Snowflake:
    \"\"\"Generate 64-bit, time-ordered IDs.\"\"\"
    EPOCH = 1704067200000  # 2024-01-01T00:00:00Z

    def __init__(self, worker_id: int = 1):
        self.worker_id = worker_id & 0x3FF  # 10 bits
        self.seq = 0
        self.last_ms = -1

    def next_id(self) -> int:
        now = int(time.time() * 1000)
        if now == self.last_ms:
            self.seq = (self.seq + 1) & 0xFFF  # 12 bits
            if self.seq == 0:
                # spin to next millisecond
                while int(time.time() * 1000) <= self.last_ms:
                    pass
                now = int(time.time() * 1000)
        else:
            self.seq = 0
        self.last_ms = now
        t = (now - self.EPOCH) & ((1 << 41) - 1)
        return (t << (10 + 12)) | (self.worker_id << 12) | self.seq


# ------------------------------------------------------------
# Data models: Posts, Comments, Memory Entries, Evidence, etc.
# ------------------------------------------------------------

@dataclass
class Post:
    id: int
    user_id: str
    content: str
    ts_ms: int
    # Root of a thread. Comments attach via parent_id relationships.


@dataclass
class Comment:
    id: int
    user_id: str
    post_id: int
    parent_id: Optional[int]  # None for top-level under post
    content: str
    ts_ms: int


@dataclass
class MemoryEntry:
    id: int
    user_id: str
    origin_kind: str            # 'post' | 'comment'
    origin_id: int
    content: str
    ts_ms: int
    confidence: float           # 0..1
    contradictions: int = 0
    status: str = "active"      # active|weakened|archived
    meta: Dict = field(default_factory=dict)


@dataclass
class EdgeEvidence:
    source_id: int     # neighbor memory id
    weight: float      # +support / -contradict (e.g., +1.0 or -1.0)
    trust: float       # 0..1 (source reliability)


@dataclass
class Reconsideration:
    memory_id: int
    old_conf: float
    new_conf: float
    decision: str      # confirm|weaken|archive
    ts_ms: int


# ------------------------------------------------------------
# Utilities: decay, reconciliation, penalties
# ------------------------------------------------------------

def temporal_decay(ts_ms: int, half_life_days: float = 30.0) -> float:
    age_days = max(0.0, (time.time()*1000 - ts_ms) / (1000*60*60*24))
    return 0.5 ** (age_days / half_life_days)


def reconcile_confidence(base: float, evidences: List[EdgeEvidence]) -> float:
    # Stable bounded update using a pseudo-logit accumulation in [-1, +1]
    score = (base - 0.5) * 2.0
    for e in evidences:
        score += e.weight * e.trust * 0.3
    # squash to [0,1]
    return max(0.0, min(1.0, 0.5 * (score + 1.0)))


def contradiction_penalty(conf: float, contradictions: int) -> float:
    return max(0.0, conf - 0.08 * contradictions)


# ------------------------------------------------------------
# Memory Store (In-memory) with Redis-like signatures
# ------------------------------------------------------------

class MemoryStore:
    \"\"\"A simple in-memory store. Swap with Redis or DB in prod.\"\"\"
    def __init__(self):
        self.posts: Dict[int, Post] = {}
        self.comments: Dict[int, Comment] = {}
        self.memories: Dict[int, MemoryEntry] = {}
        self.user_index: Dict[str, List[int]] = defaultdict(list)  # user_id -> [memory_ids]
        self.recons: List[Reconsideration] = []

    # --- Post & Comment ingestion ---
    def save_post(self, p: Post):
        self.posts[p.id] = p

    def save_comment(self, c: Comment):
        self.comments[c.id] = c

    # --- Memory entry CRUD ---
    def save_entry(self, m: MemoryEntry) -> None:
        self.memories[m.id] = m
        if m.id not in self.user_index[m.user_id]:
            self.user_index[m.user_id].append(m.id)

    def get_entries(self, user_id: str, start_idx: int = 0) -> List[MemoryEntry]:
        ids = self.user_index.get(user_id, [])
        return [self.memories[i] for i in ids]

    def save_reconsideration(self, r: Reconsideration) -> None:
        self.recons.append(r)

    # --- Thread traversal helpers ---
    def get_post(self, post_id: int) -> Optional[Post]:
        return self.posts.get(post_id)

    def get_comment(self, comment_id: int) -> Optional[Comment]:
        return self.comments.get(comment_id)

    def get_thread_comments(self, post_id: int) -> List[Comment]:
        return [c for c in self.comments.values() if c.post_id == post_id]


# ------------------------------------------------------------
# Consensus Graph (Vector DB or link-based). Stubbed here.
# ------------------------------------------------------------

class ConsensusGraph:
    def __init__(self, mem: MemoryStore):
        self.mem = mem

    def neighbors(self, mem_id: int, k: int = 6) -> List[EdgeEvidence]:
        \"\"\"Return related memories as evidence.
        Stub: sample random neighbors with mixed support/contradiction.
        In prod: replace with vector similarity + link graph signals.
        \"\"\"
        all_ids = list(self.mem.memories.keys())
        all_ids = [i for i in all_ids if i != mem_id]
        random.shuffle(all_ids)
        chosen = all_ids[:k]

        evidences = []
        for nid in chosen:
            weight = random.choice([+1.0, +1.0, -1.0])  # mostly supporting in demo
            trust  = random.uniform(0.6, 0.95)
            evidences.append(EdgeEvidence(source_id=nid, weight=weight, trust=trust))
        return evidences


# ------------------------------------------------------------
# Comment threading + trace
# ------------------------------------------------------------

def build_comment_tree(comments: List[Comment]) -> Dict[Optional[int], List[Comment]]:
    by_parent: Dict[Optional[int], List[Comment]] = defaultdict(list)
    for c in sorted(comments, key=lambda x: x.ts_ms):
        by_parent[c.parent_id].append(c)
    return by_parent


def trace_thread(mem: MemoryStore, post_id: int) -> None:
    post = mem.get_post(post_id)
    if not post:
        LOGGER.warning(f"[trace] Post {post_id} not found.")
        return

    LOGGER.info(f"[trace] Post {post.id} by {post.user_id} @ {post.ts_ms}: {post.content}")
    comments = mem.get_thread_comments(post_id)
    tree = build_comment_tree(comments)

    def walk(parent_id: Optional[int], depth: int = 0):
        for c in tree.get(parent_id, []):
            indent = "  " * depth
            LOGGER.info(f\"{indent}- Comment {c.id} by {c.user_id} @ {c.ts_ms}: {c.content}\")
            walk(c.id, depth + 1)

    walk(None)


# ------------------------------------------------------------
# Reconsideration Engine
# ------------------------------------------------------------

def reconsider_user(mem: MemoryStore, cg: ConsensusGraph, user_id: str) -> List[Reconsideration]:
    entries = mem.get_entries(user_id, start_idx=0x0)

    def priority(e: MemoryEntry):
        # sort & reconsider
        return (
            -(e.confidence),              # higher first
            e.contradictions,             # fewer first
            -temporal_decay(e.ts_ms),     # newer favored (inverse decay)
            -e.ts_ms                      # tiebreaker: newer first
        )
    entries.sort(key=priority)

    results: List[Reconsideration] = []
    for e in entries:
        LOGGER.info(f\"[reconsider] -> {e.id} ({e.origin_kind}:{e.origin_id}) conf={e.confidence:.3f} status={e.status}\")
        base = e.confidence * temporal_decay(e.ts_ms)
        ev = cg.neighbors(e.id)
        conf = reconcile_confidence(base, ev)
        conf = contradiction_penalty(conf, e.contradictions)

        if   conf >= 0.80: decision, status = "confirm", "active"
        elif conf >= 0.50: decision, status = "weaken",  "weakened"
        else:              decision, status = "archive", "archived"

        rec = Reconsideration(
            memory_id=e.id, old_conf=e.confidence,
            new_conf=conf, decision=decision, ts_ms=int(time.time()*1000)
        )
        e.confidence, e.status = conf, status
        mem.save_entry(e)
        mem.save_reconsideration(rec)
        results.append(rec)

        LOGGER.info(f\"[reconsider] <- {e.id} new_conf={conf:.3f} decision={decision} status={status}\")

    return results


def reconsider_many(mem: MemoryStore, cg: ConsensusGraph, user_ids: Iterable[str], max_workers: int = 4) -> Dict[str, List[Reconsideration]]:
    out: Dict[str, List[Reconsideration]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut2uid = {pool.submit(reconsider_user, mem, cg, uid): uid for uid in user_ids}
        for fut in as_completed(fut2uid):
            uid = fut2uid[fut]
            try:
                out[uid] = fut.result()
            except Exception as e:
                LOGGER.exception(f\"[reconsider_many] user {uid} failed: {e}\")
    return out


# ------------------------------------------------------------
# Ingestion helpers
# ------------------------------------------------------------

def ingest_post(mem: MemoryStore, snow: Snowflake, user_id: str, content: str) -> Post:
    p = Post(id=snow.next_id(), user_id=user_id, content=content, ts_ms=int(time.time()*1000))
    mem.save_post(p)
    # Also create a MemoryEntry for the post content
    m = MemoryEntry(
        id=snow.next_id(), user_id=user_id, origin_kind="post", origin_id=p.id,
        content=content, ts_ms=p.ts_ms, confidence=0.70, meta={"anchor": 0x0}
    )
    mem.save_entry(m)
    return p


def ingest_comment(mem: MemoryStore, snow: Snowflake, user_id: str, post_id: int, content: str, parent_id: Optional[int] = None) -> Comment:
    c = Comment(id=snow.next_id(), user_id=user_id, post_id=post_id, parent_id=parent_id, content=content, ts_ms=int(time.time()*1000))
    mem.save_comment(c)
    # Also create a MemoryEntry for the comment content
    m = MemoryEntry(
        id=snow.next_id(), user_id=user_id, origin_kind="comment", origin_id=c.id,
        content=content, ts_ms=c.ts_ms, confidence=0.65, meta={"anchor": 0x0}
    )
    mem.save_entry(m)
    return c


# ------------------------------------------------------------
# Demo CLI
# ------------------------------------------------------------

def run_demo(seed: int = 42, users: int = 3, posts_per_user: int = 2, comments_per_post: int = 4):
    random.seed(seed)
    mem = MemoryStore()
    snow = Snowflake(worker_id=1)

    user_ids = [f\"user_{i}\" for i in range(users)]
    contents = [
        "Q: Is temporal decay too aggressive?",
        "Design: Reconsideration thresholds at 0.8/0.5",
        "Spec: Add contradiction counter",
        "Note: Cross-link to consensus evidence",
        "Meta: Anchor at 0x0 for head list",
        "Perf: Sort priority favors recency"
    ]

    # Ingest posts and comments (build threads)
    for uid in user_ids:
        for _ in range(posts_per_user):
            p = ingest_post(mem, snow, uid, random.choice(contents))
            # Build a small comment tree
            parents = [None]  # start with top-level
            for j in range(comments_per_post):
                parent = random.choice(parents)
                c = ingest_comment(mem, snow, random.choice(user_ids), p.id, f\"Reply {j} to {p.id}\", parent_id=parent)
                parents.append(c.id)  # allow replies to replies

    # Trace a random thread
    any_post_id = random.choice(list(mem.posts.keys()))
    LOGGER.info(f\"========== TRACE THREAD (post {any_post_id}) ==========\")
    trace_thread(mem, any_post_id)

    # Reconsider in parallel
    cg = ConsensusGraph(mem)
    LOGGER.info(\"========== RECONSIDERATION (parallel) ==========\")
    results = reconsider_many(mem, cg, user_ids, max_workers=min(4, users))

    # Summarize
    LOGGER.info(\"========== SUMMARY ==========\")
    for uid, recs in results.items():
        confirms = sum(1 for r in recs if r.decision == "confirm")
        weakens  = sum(1 for r in recs if r.decision == "weaken")
        archives = sum(1 for r in recs if r.decision == "archive")
        LOGGER.info(f\"{uid}: total={len(recs)} confirm={confirms} weaken={weakens} archive={archives}\")


def main():
    parser = argparse.ArgumentParser(description=\"Snowflake reconsideration + comment threading trace\")
    parser.add_argument(\"--demo\", action=\"store_true\", help=\"Run a local demo with synthetic data.\")
    parser.add_argument(\"--seed\", type=int, default=42, help=\"Random seed for demo.\")
    parser.add_argument(\"--users\", type=int, default=3, help=\"Number of users to simulate in demo.\")
    parser.add_argument(\"--posts-per-user\", type=int, default=2, help=\"Posts per user in demo.\")
    parser.add_argument(\"--comments-per-post\", type=int, default=4, help=\"Comments per post in demo.\")
    parser.add_argument(\"--log-level\", default=\"INFO\", help=\"Logging level (DEBUG, INFO, WARNING, ERROR)\")
    args = parser.parse_args()

    LOGGER.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    if args.demo:
        run_demo(seed=args.seed, users=args.users, posts_per_user=args.posts_per_user, comments_per_post=args.comments_per_post)
    else:
        print(\"Nothing to do. Use --demo to run a quick simulation.\")


if __name__ == \"__main__\":
    main()
''')

out_path = Path("/mnt/data/snowflake_reconsideraiton_comment_threading_trace.py")
out_path.write_text(code, encoding="utf-8")

print("Created:", str(out_path))
print("\\nQuick start:")
print("  python /mnt/data/snowflake_reconsideraiton_comment_threading_trace.py --demo --users 3 --posts-per-user 2 --comments-per-post 4 --log-level INFO")
