/*
 * 工单编号：人工智能NLP-Agent数字人项目-医疗智能体-实时语音识别、翻译与会议概要
 *
 * AudioWorklet Processor —— 从麦克风流中累积 40ms PCM 16k mono 帧，
 * 每帧 640 samples(Int16) = 1280 bytes 通过 port.postMessage 传出。
 *
 * 讯飞 IAT 要求：audio/L16; rate=16000; encoding=raw; 单声道；每帧 40ms。
 * AudioContext 用 sampleRate:16000 构造，本 worklet 直接收到 16k Float32,
 * 不需要重采样。
 */

class PcmFramer extends AudioWorkletProcessor {
  static get parameterDescriptors() { return []; }

  constructor() {
    super();
    this.frameSize = 640;                // 40ms @ 16kHz
    this.buffer   = new Float32Array(this.frameSize);
    this.offset   = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch = input[0];
    if (!ch || ch.length === 0) return true;

    for (let i = 0; i < ch.length; i++) {
      this.buffer[this.offset++] = ch[i];
      if (this.offset === this.frameSize) {
        // Float32 → Int16
        const pcm = new Int16Array(this.frameSize);
        let sumSq = 0;
        for (let j = 0; j < this.frameSize; j++) {
          const s = Math.max(-1, Math.min(1, this.buffer[j]));
          pcm[j] = s < 0 ? s * 0x8000 : s * 0x7FFF;
          sumSq += this.buffer[j] * this.buffer[j];
        }
        const rms = Math.sqrt(sumSq / this.frameSize);
        // 帧和音量一起发；transferable 减少拷贝
        this.port.postMessage(
          { pcm: pcm.buffer, rms },
          [pcm.buffer]
        );
        this.offset = 0;
      }
    }
    return true;
  }
}

registerProcessor("pcm-framer", PcmFramer);
