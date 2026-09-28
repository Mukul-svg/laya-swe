# Task 3: Iterative Bug Localization

### Turn 1 Prompt:
Run pytest on `tasks/task_3_iterative_debugging/tests/test_lru_race.py` to see the failure. Analyze `cache/lru_manager.py` to understand why keys are prematurely evicted.

### Turn 2 Prompt (Hypothesis & Diagnostic):
Explain what is wrong with the eviction order logic when `put()` evicts keys.

### Turn 3 Prompt (Fix & Verify):
Fix the bug in `cache/lru_manager.py` so that least recently used keys are evicted from the front of the queue (`self.order.pop(0)` or `collections.OrderedDict`), and ensure all tests pass.
