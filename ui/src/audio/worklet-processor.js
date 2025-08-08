class PCMBridge extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.ptr = 0;
    this.current = null;
    this.port.onmessage = e => {
      if (e.data?.type === 'push') {
        this.queue.push(e.data.buffer);
      }
    }
  }
  process(inputs, outputs) {
    const out = outputs[0][0];
    let i = 0;
    while (i < out.length) {
      if (!this.current || this.ptr >= this.current.length) {
        this.current = this.queue.shift() || null;
        this.ptr = 0;
        if (!this.current) { // fill silence
          for (; i < out.length; i++) out[i] = 0;
          return true;
        }
      }
      const rem = Math.min(out.length - i, this.current.length - this.ptr);
      out.set(this.current.subarray(this.ptr, this.ptr + rem), i);
      i += rem; this.ptr += rem;
    }
    return true;
  }
}
registerProcessor('pcm-bridge', PCMBridge);
