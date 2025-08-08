# tts_ws.py
import asyncio, json, base64, threading
import websockets

class WSBroadcaster:
    def __init__(self, host="0.0.0.0", port=8765):
        self.host, self.port = host, port
        self.clients = set()
        self.loop = asyncio.new_event_loop()
        self.server = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    async def _handler(self, ws):
        self.clients.add(ws)
        try:
            async for _ in ws:
                pass
        finally:
            self.clients.discard(ws)

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.server = self.loop.run_until_complete(websockets.serve(self._handler, self.host, self.port))
        self.loop.run_forever()

    def _broadcast(self, obj):
        if not self.clients: return
        data = json.dumps(obj)
        asyncio.run_coroutine_threadsafe(self._broadcast_async(data), self.loop)

    async def _broadcast_async(self, data):
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    # API
    def tts_begin(self, sr:int, msg_id:str):
        self._broadcast({"type":"tts_begin","sr":sr,"id":msg_id})

    def tts_chunk(self, msg_id:str, ts:float, pcm_f32):
        # pcm_f32: numpy float32 array
        b = base64.b64encode(pcm_f32.tobytes()).decode("ascii")
        self._broadcast({"type":"tts_chunk","id":msg_id,"ts":ts,"pcm":b})

    def tts_end(self, msg_id:str):
        self._broadcast({"type":"tts_end","id":msg_id})

    def state(self, value:str):
        self._broadcast({"type":"state","value":value})

    def emotion(self, value:str):
        self._broadcast({"type":"emotion","value":value})

WS = WSBroadcaster()  # singleton import
