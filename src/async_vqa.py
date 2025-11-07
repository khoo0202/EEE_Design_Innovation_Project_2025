import threading, queue, time


class AsyncVQAWorker:
    def __init__(self, vqa_client, callback, maxsize=32, timeout_s=10):
        self.vqa = vqa_client
        self.callback = callback
        self.queue = queue.Queue(maxsize=maxsize)
        self.timeout_s = timeout_s
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self._stop_event.is_set():
            try:
                tid, frames = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                vqa = self.vqa.classify_frames(frames)
                self.callback(tid, vqa)
            except Exception as e:
                print(f"[AsyncVQAWorker] Error: {e}")
            finally:
                self.queue.task_done()

    def submit(self, track_id, frames):
        try:
            self.queue.put_nowait((track_id, frames))
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
                self.queue.put_nowait((track_id, frames))
            except queue.Full:
                pass

    def stop(self):
        self._stop_event.set()
        try:
            self.thread.join(timeout=self.timeout_s)
        except Exception:
            pass
