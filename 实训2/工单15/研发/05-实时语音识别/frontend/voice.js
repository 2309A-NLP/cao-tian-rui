/*
 * 工单编号：人工智能NLP-Agent数字人项目-医疗智能体-实时语音识别、翻译与会议概要
 *
 * 语音助手主控 —— WebSocket 客户端 + AudioWorklet PCM 采集 + 事件渲染。
 *
 * 连接后端 /ws/stream:
 *   ↑ init 帧(JSON)   {type:"init", lang:"zh_cn"}
 *   ↑ PCM binary      每 40ms 一帧, 16kHz 16bit mono
 *   ↑ stop 帧(JSON)   {type:"stop"}
 *   ↓ 接收事件         started/transcription/sentence_end/translation/completed/error/log
 *
 * 停止后自动切"会议纪要" Tab, 拉 /summary 或轮询 /poll 填充。
 */

(() => {
  "use strict";

  const $  = (id) => document.getElementById(id);
  const $$ = (sel) => document.querySelectorAll(sel);
  const API_BASE = `${location.protocol}//${location.host}`;
  const WS_PROTO = location.protocol === "https:" ? "wss" : "ws";
  const WS_URL   = `${WS_PROTO}://${location.host}/ws/stream`;

  // ────────── 状态 ──────────
  // L3：静音帧（640×Int16 全零 = 40ms 静音），用于暂停时维持讯飞 VAD 活跃
  const SILENCE_FRAME = new Int16Array(640).buffer;

  const state = {
    ws: null,
    audioCtx: null,
    workletNode: null,
    mediaStream: null,
    micSource: null,
    recording: false,
    paused: false,
    silenceTimer: null,     // L3：暂停保活定时器
    taskId: null,
    startedAt: null,
    interimEl: null,
    sentenceCount: 0,       // 用于 speaker 交替显示
    transcript: [],         // 累积句
    pollTimer: null,
    modeInfo: { mock_mode: true, asr: "mock" },
  };

  // ────────── 日志 ──────────
  function log(msg, type = "") {
    const box = $("logBox"); if (!box) return;
    const line = document.createElement("div");
    line.className = "log-line" + (type ? " " + type : "");
    line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    box.appendChild(line);
    box.scrollTop = box.scrollHeight;
  }

  // 会议纪要 Tab 上的简短 toast（不依赖 logBox）
  function toastSummary(msg, type = "") {
    const el = $("summaryToast");
    if (!el) return;
    el.textContent = msg;
    el.className = "summary-toast" + (type ? " " + type : "");
    el.style.display = "";
    clearTimeout(el._timer);
    el._timer = setTimeout(() => { el.style.display = "none"; }, 3000);
  }

  const escHtml = (s) => String(s ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  function formatMs(ms) {
    const s = Math.floor((ms || 0) / 1000);
    const m = Math.floor(s / 60);
    return `${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
  }

  // ────────── Tab 切换 ──────────
  function switchTab(target) {
    $$(".tab").forEach((t) =>
      t.classList.toggle("active", t.dataset.panel === target)
    );
    $$(".panel").forEach((p) =>
      p.classList.toggle("active", p.id === `panel-${target}`)
    );
  }
  $$(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.panel)));

  // 语言交换
  document.querySelector(".swap-btn")?.addEventListener("click", () => {
    const a = $("langSrc"), b = $("langTgt");
    const va = a.value; a.value = b.value; b.value = va;
    log("语言方向已交换", "info");
  });

  // ────────── 麦克风:开始 ──────────
  async function startRecording() {
    if (state.recording) return;
    const lang = $("langSrc").value;

    // 清空静态占位内容
    $("transcriptContent").innerHTML = "";
    state.transcript = [];
    state.sentenceCount = 0;
    state.startedAt = Date.now();

    // 1) 请求麦克风
    try {
      state.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      log("麦克风权限被拒绝: " + e.message, "err");
      return;
    }

    // 2) 建 WS
    log("正在连接 WebSocket…", "info");
    try {
      state.ws = new WebSocket(WS_URL);
    } catch (e) {
      log("WS 构造失败: " + e.message, "err");
      return;
    }

    state.ws.onopen = async () => {
      log("WebSocket 已连接, 发送 init…", "info");
      // L2：携带前端选择的翻译目标语言，后端用于实际翻译（而非读 .env 默认值）
      const translate_lang = $("langTgt").value;
      state.ws.send(JSON.stringify({ type: "init", lang, translate_lang }));
      await startAudioPipeline();
      setRecordingUi(true);
    };
    state.ws.onmessage = (e) => {
      try { handleWsEvent(JSON.parse(e.data)); }
      catch (err) { log("解析 WS 消息失败: " + err.message, "err"); }
    };
    state.ws.onerror = () => log("WebSocket 错误", "err");
    state.ws.onclose = () => {
      log("WebSocket 已断开", "info");
      stopAudioPipeline();
      setRecordingUi(false);
    };
  }

  // ────────── 音频管道:AudioWorklet 抽帧 ──────────
  async function startAudioPipeline() {
    state.audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });

    try {
      await state.audioCtx.audioWorklet.addModule("/static/pcm-worklet.js");
    } catch (e) {
      log("AudioWorklet 加载失败: " + e.message, "err");
      throw e;
    }

    state.micSource = state.audioCtx.createMediaStreamSource(state.mediaStream);
    state.workletNode = new AudioWorkletNode(state.audioCtx, "pcm-framer");
    state.workletNode.port.onmessage = (ev) => {
      if (state.paused) return;
      const { pcm, rms } = ev.data;
      // 音量条
      $("volumeBar").style.width = Math.min(rms * 500, 100) + "%";
      // 发送到后端(二进制)
      if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(pcm);
      }
    };

    // 连接图: mic → worklet → (silent) → destination
    state.micSource.connect(state.workletNode);
    // 静音接到 destination 保持图激活(不接则 Chrome 会挂起 worklet)
    state.workletNode.connect(state.audioCtx.destination);
    log(`音频管道启动 · 采样率=${state.audioCtx.sampleRate}Hz · 帧=40ms`, "ok");
  }

  function stopAudioPipeline() {
    clearInterval(state.silenceTimer);  // L3：停止时清除保活定时器
    state.silenceTimer = null;
    state.paused = false;
    try { state.workletNode?.disconnect(); } catch {}
    try { state.micSource?.disconnect(); } catch {}
    try { state.audioCtx?.close(); } catch {}
    state.mediaStream?.getTracks().forEach((t) => t.stop());
    state.workletNode = state.micSource = state.audioCtx = state.mediaStream = null;
    $("volumeBar").style.width = "0%";
  }

  // ────────── 麦克风:停止 ──────────
  function stopRecording() {
    if (!state.recording) return;
    log("发送 stop 指令并结束录音…", "info");
    try {
      if (state.ws?.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify({ type: "stop" }));
      }
    } catch (e) { log("发送 stop 失败: " + e.message, "err"); }
    setRecordingUi(false, "处理中,等待摘要…");
  }

  // ────────── UI 切换 ──────────
  function setRecordingUi(rec, statusOverride) {
    state.recording = rec;
    const micBtn = $("micBtn"), micIcon = $("micIcon"), st = $("statusText");
    micBtn.classList.toggle("recording", rec);
    micIcon.textContent = rec ? "⏹️" : "🎤";
    st.textContent = statusOverride
      ?? (rec ? "🔴 正在录音… 实时识别中" : "点击麦克风开始录音,支持实时语音识别与翻译");
    st.classList.toggle("recording", rec);
    $("btnStop").disabled = !rec;
    $("btnPause").disabled = !rec;
    $("btnPause").textContent = state.paused ? "▶️ 继续" : "⏸️ 暂停";
  }

  // ────────── WS 事件处理 ──────────
  function handleWsEvent(msg) {
    switch (msg.type) {
      case "started":
        state.taskId = msg.task_id;
        log(`任务已创建: ${msg.task_id}`, "ok");
        break;
      case "log":
        log(`[server] ${msg.message}`, "info");
        break;
      case "transcription":
        renderInterim(msg.text, msg.speaker);
        break;
      case "sentence_end":
        renderSentenceEnd(msg);
        break;
      case "translation":
        renderTranslation(msg);
        break;
      case "completed":
        log(`任务完成: ${msg.task_id}`, "ok");
        onSessionCompleted();
        break;
      case "error":
        log(`服务端错误: ${msg.message}`, "err");
        break;
      default:
        log("未知事件: " + JSON.stringify(msg), "info");
    }
  }

  function renderInterim(text, speaker) {
    const box = $("transcriptContent");
    if (!state.interimEl) {
      state.interimEl = document.createElement("div");
      state.interimEl.className = "transcript-line interim";
      box.appendChild(state.interimEl);
    }
    state.interimEl.innerHTML = `⟳ ${escHtml(text)}`;
    box.scrollTop = box.scrollHeight;
  }

  function renderSentenceEnd(msg) {
    if (state.interimEl) { state.interimEl.remove(); state.interimEl = null; }
    const speaker = msg.speaker || "发言人";
    // 交替医患角色(讯飞不带说话人分离,靠句序模拟)
    const isDoc = (state.sentenceCount % 2) === 0;
    const chipCls = isDoc ? "doc" : "pat";
    const chipTxt = isDoc ? "医生" : "患者";

    const el = document.createElement("div");
    el.className = "transcript-line new";
    el.dataset.idx = state.sentenceCount;
    el.innerHTML = `
      <span class="speaker-chip ${chipCls}">${chipTxt}</span>
      <span class="line-text">${escHtml(msg.text)}</span>
      <div class="translation" data-slot="translation" style="display:none;"></div>
    `;
    $("transcriptContent").appendChild(el);
    $("transcriptContent").scrollTop = $("transcriptContent").scrollHeight;
    state.transcript.push({ text: msg.text, speaker: chipTxt });
    state.sentenceCount++;
  }

  function renderTranslation(msg) {
    const box = $("transcriptContent");
    // 找到对应句子(按 sentence_index)
    const idx = msg.sentence_index;
    const line = box.querySelector(`.transcript-line[data-idx="${idx}"]`);
    if (!line) return;
    const slot = line.querySelector('[data-slot="translation"]');
    if (slot) {
      slot.style.display = "";
      slot.textContent = `🌐 ${msg.text}`;
    }
  }

  // ────────── 会话完成:自动切纪要 + 拉摘要 ──────────
  async function onSessionCompleted() {
    setRecordingUi(false, "录音完成");
    stopAudioPipeline();
    try { state.ws?.close(); } catch {}
    $("meetingTime").textContent =
      `会议/问诊时间: ${new Date(state.startedAt).toLocaleString()} — ${new Date().toLocaleTimeString()}`;

    // 自动切 Tab
    switchTab("summary");
    log("已切换到会议纪要 Tab,拉取摘要中…", "info");

    // 触发首次 poll(3s 一次, 最多 30 次 = 90s), 收到 COMPLETED 就停
    startPolling();
  }

  function startPolling() {
    if (!state.taskId) return;
    if (state.pollTimer) clearInterval(state.pollTimer);
    let attempts = 0;

    async function tick() {
      attempts++;
      if (attempts > 30) { clearInterval(state.pollTimer); log("轮询超时", "err"); return; }
      try {
        const r = await fetch(`${API_BASE}/api/session/${state.taskId}/poll`);
        const data = await r.json();
        log(`轮询 #${attempts}: TaskStatus=${data.TaskStatus}`, "info");
        renderSummaryData(data);
        if (data.TaskStatus === "COMPLETED" || data.TaskStatus === "stopped") {
          clearInterval(state.pollTimer);
          log("已获取到最终结果", "ok");
        }
      } catch (e) {
        log(`轮询失败: ${e.message}`, "err");
      }
    }
    tick(); // 立即拉一次
    state.pollTimer = setInterval(tick, 3000);
  }

  function renderSummaryData(data) {
    const result = data.Result || data.result || {};
    const sum = result.Summarization || result.summarization || {};
    const chapters = result.Chapters || result.chapters || [];

    // 全文摘要
    if (sum.Paragraph || sum.paragraph) {
      $("sumParagraph").textContent = sum.Paragraph || sum.paragraph;
    }

    // 章节速览
    if (chapters.length) {
      $("sumChapters").innerHTML = chapters.map((c) => {
        const t = c.Headline || c.headline || "-";
        const body = c.Summary || c.summary || "";
        const time = c.BeginTime != null ? formatMs(c.BeginTime) : "-";
        return `<div class="chapter-row">
          <div class="ch-title">${escHtml(t)}</div>
          <div class="ch-body">${escHtml(time)} · ${escHtml(body)}</div>
        </div>`;
      }).join("");
    }

    // 待办
    const todos = sum.ActionItems || sum.action_items || [];
    if (todos.length) {
      $("sumTodos").innerHTML = todos.map((t) => {
        const txt = typeof t === "string" ? t : JSON.stringify(t);
        return `<div class="todo-item"><div class="todo-checkbox"></div><span>${escHtml(txt)}</span></div>`;
      }).join("");
      // 重新绑定勾选
      $$(".todo-checkbox").forEach((cb) =>
        cb.addEventListener("click", () => cb.classList.toggle("checked"))
      );
    }

    // 关键词
    const kws = sum.KeyInformation || sum.key_information || [];
    if (kws.length) {
      $("sumKeywords").innerHTML = kws.map((k) => {
        const txt = typeof k === "string" ? k : JSON.stringify(k);
        return `<span class="keyword-tag">${escHtml(txt)}</span>`;
      }).join("");
    }

    // 问答: 若后端未返回, 保留静态占位
  }

  // ────────── 手动动作 ──────────
  $("micBtn").addEventListener("click", () => {
    if (state.recording) stopRecording(); else startRecording();
  });

  $("btnStop").addEventListener("click", () => stopRecording());

  $("btnPause").addEventListener("click", () => {
    state.paused = !state.paused;
    setRecordingUi(state.recording, state.paused ? "⏸️ 已暂停" : "🔴 正在录音…");
    log(state.paused ? "已暂停音频推送" : "已恢复音频推送", "info");

    // L3：暂停时每 2 秒发一帧静音，防止讯飞 VAD 10s 超时断开连接
    if (state.paused) {
      state.silenceTimer = setInterval(() => {
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
          state.ws.send(SILENCE_FRAME.slice(0));  // slice(0) 复制一份，避免 Transferable 问题
        }
      }, 2000);
    } else {
      clearInterval(state.silenceTimer);
      state.silenceTimer = null;
    }
  });

  $("btnPoll").addEventListener("click", () => {
    if (!state.taskId) { toastSummary("无 task_id，请先录音一次", "err"); return; }
    toastSummary("正在拉取最新摘要…", "info");
    log("手动触发轮询", "info");
    startPolling();
  });

  $("btnConsult").addEventListener("click", async () => {
    if (!state.taskId) { toastSummary("无 task_id，请先录音一次", "err"); return; }
    $("consultBox").textContent = "正在向工单12健康咨询 Agent 发送请求…";
    $("consultBox").scrollIntoView({ behavior: "smooth", block: "center" });
    try {
      const r = await fetch(`${API_BASE}/api/session/${state.taskId}/consult`, { method: "POST" });
      const data = await r.json();
      const reply = data.health_reply?.reply || data.health_reply || JSON.stringify(data, null, 2);
      $("consultBox").textContent = typeof reply === "string" ? reply : JSON.stringify(reply, null, 2);
      toastSummary("健康建议已获取", "ok");
      log("健康建议已获取", "ok");
    } catch (e) {
      $("consultBox").textContent = "调用失败: " + e.message;
      toastSummary("工单12调用失败", "err");
      log("工单12调用失败: " + e.message, "err");
    }
  });

  $("btnCopy").addEventListener("click", async () => {
    const text = [
      "全文摘要:", $("sumParagraph").innerText, "",
      "章节速览:", $("sumChapters").innerText, "",
      "待办事项:", $("sumTodos").innerText, "",
      "关键词:", $("sumKeywords").innerText,
    ].join("\n");
    try { await navigator.clipboard.writeText(text); log("已复制", "ok"); }
    catch (e) { log("复制失败: " + e.message, "err"); }
  });

  $("btnExport").addEventListener("click", async () => {
    if (!state.taskId) { log("无 task_id", "err"); return; }
    const r = await fetch(`${API_BASE}/api/session/${state.taskId}/summary`);
    const data = await r.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `meeting-${state.taskId}.json`;
    a.click();
    URL.revokeObjectURL(url);
    log("已导出 JSON", "ok");
  });

  $("btnSettings").addEventListener("click", () => {
    alert("引擎切换请修改 backend/.env 中 ASR_ENGINE 后重启服务。\n" +
          "翻译目标语言: TRANSLATE_LANGUAGES=en,ja");
  });

  // 待办勾选(初始静态项)
  $$(".todo-checkbox").forEach((cb) =>
    cb.addEventListener("click", () => cb.classList.toggle("checked"))
  );

  // ────────── 健康探针 ──────────
  fetch(`${API_BASE}/health`)
    .then((r) => r.json())
    .then((data) => {
      state.modeInfo = data;
      const badge = $("modeBadge");
      const modeLabel = data.mock_mode
        ? "MOCK 模式(内置对话)"
        : `${data.asr.toUpperCase()} 模式`;
      badge.textContent = modeLabel;
      badge.className = "badge " + (data.mock_mode ? "mock" : "");
      log(`后端就绪 · ASR=${data.asr} · MOCK=${data.mock_mode} · 翻译=${data.translate_enabled}`, "ok");
    })
    .catch((e) => {
      const badge = $("modeBadge");
      badge.textContent = "离线";
      badge.style.background = "#FEE2E2";
      badge.style.color = "#991B1B";
      log("后端离线: " + e.message, "err");
    });
})();
