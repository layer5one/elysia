class PCMBridge extends AudioWorkletProcessor {
  constructor(){ super(); this.q=[]; this.ptr=0; this.cur=null;
    this.port.onmessage = e => { if(e.data?.type==='push') this.q.push(e.data.buf); };
  }
  process(inputs, outputs){
    const out = outputs[0][0]; let i=0;
    while(i<out.length){
      if(!this.cur || this.ptr>=this.cur.length){ this.cur=this.q.shift()||null; this.ptr=0; if(!this.cur){ out.fill(0); return true; } }
      const n = Math.min(out.length-i, this.cur.length-this.ptr);
      out.set(this.cur.subarray(this.ptr, this.ptr+n), i); i+=n; this.ptr+=n;
    }
    return true;
  }
}
registerProcessor('pcm-bridge', PCMBridge);
