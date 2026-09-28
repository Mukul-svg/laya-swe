"""LRU Cache Implementation with a subtle eviction bug."""

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.order = []

    def get(self, key: str):
        if key not in self.cache:
            return None
        # BUG: It updates order, but if key is accessed multiple times, it appends duplicates
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]

    def put(self, key: str, value: any):
        if key in self.cache:
            self.cache[key] = value
            self.order.remove(key)
            self.order.append(key)
            return

        # Evict oldest if full
        if len(self.cache) >= self.capacity:
            # BUG: Pops from index -1 (newest) instead of index 0 (oldest)!
            oldest = self.order.pop(0)
            if oldest in self.cache:
                del self.cache[oldest]

        self.cache[key] = value
        self.order.append(key)
