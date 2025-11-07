from collections import deque, Counter

class TemporalSmoother:
    def __init__(self, window=7, min_persist=3):
        self.win = window
        self.min_persist = min_persist
        self.buf = {}   # tid -> deque[str]
        self.pbuf = {}  # tid -> deque[dict] (prob dicts)

    def update(self, tid: int, label: str, probs: dict):
        dq = self.buf.setdefault(tid, deque(maxlen=self.win))
        pq = self.pbuf.setdefault(tid, deque(maxlen=self.win))
        dq.append(label)
        pq.append(probs)

    def read(self, tid: int):
        if tid not in self.buf: return None, 0.0, {}
        dq, pq = self.buf[tid], self.pbuf[tid]
        if len(dq) < self.min_persist: return None, 0.0, {}
        # majority vote
        cnt = Counter(dq)
        label, votes = cnt.most_common(1)[0]
        if votes < self.min_persist:
            return None, 0.0, {}
        # averaged probs
        keys = set().union(*pq)
        avg = {k: sum(d.get(k,0.0) for d in pq)/len(pq) for k in keys}
        return label, avg.get(label, 0.0), avg

    def clear(self, tid: int):
        self.buf.pop(tid, None)
        self.pbuf.pop(tid, None)
