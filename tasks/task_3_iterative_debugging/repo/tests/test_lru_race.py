"""Ground truth validation test for Task 3: LRU Eviction Fix."""
import pytest
import sys
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(cur_dir)
for candidate in [parent_dir, os.path.join(parent_dir, "repo"), os.getcwd()]:
    if candidate not in sys.path and os.path.isdir(candidate):
        sys.path.insert(0, candidate)
repo_dir = parent_dir if os.path.basename(parent_dir) == "repo" else os.path.join(parent_dir, "repo")

from cache.lru_manager import LRUCache

def test_lru_eviction_behavior():
    """Verify that LRU strictly evicts the least recently used key, not the newest."""
    cache = LRUCache(capacity=2)
    cache.put("k1", 1)
    cache.put("k2", 2)
    
    # Access k1 to make k2 the least recently used
    assert cache.get("k1") == 1
    
    # Put k3; should evict k2, keeping k1 and k3
    cache.put("k3", 3)
    
    assert cache.get("k1") == 1, "k1 should still exist (it was recently accessed)"
    assert cache.get("k2") is None, "k2 should have been evicted as the LRU element"
    assert cache.get("k3") == 3, "k3 was just added and should exist"

def test_lru_capacity_overflow():
    """Verify capacity bound across multiple inserts."""
    cache = LRUCache(capacity=3)
    for i in range(10):
        cache.put(f"key_{i}", i)
    assert len(cache.cache) <= 3
