/* =====================================================================
   Agent22 主逻辑
   - Tab 切换（医疗 / 文旅 / 教育）
   - 用户ID 管理（每个 domain 独立保存）
   - 对话发送、消息渲染、打字动画
   - 右侧记忆面板（显示全部记忆 + 高亮本轮召回）
   - 模态框展示本轮召回详情
   - 清空记忆
===================================================================== */

const DOMAINS = {
  medical:   { label: '医疗助理', emoji: '🏥', color: '#e74c6f', desc: '健康咨询 · 复诊记忆 · 用药安全' },
  travel:    { label: '文旅规划', emoji: '✈️', color: '#27ae60', desc: '偏好记忆 · 个性化推荐 · 行程规划' },
  education: { label: '教育辅导', emoji: '📚', color: '#f39c12', desc: '学习进度 · 薄弱点强化 · 因材施教' },
};

// ── 状态 ────────────────────────────────────────────────────────────
let currentDomain = 'medical';
const sessions   = {};   // domain -> session_id
const userIds    = {};   // domain -> user_id
const histories  = {};   // domain -> [{role, content, recalled}]
let lastRecalled = [];   // 上一次回复的 recalled 列表（供模态框）

// ── DOM 引用 ─────────────────────────────────────────────────────────
const messagesWrap  = document.getElementById('messages');
const textarea      = document.getElementById('user-input');
const sendBtn       = document.getElementById('send-btn');
const userIdInput   = document.getElementById('user-id-input');
const clearBtn      = document.getElementById('clear-btn');
const refreshBtn    = document.getElementById('refresh-memory-btn');
const memoryList    = document.getElementById('memory-list');
const domainBadge   = document.getElementById('domain-badge');
const domainDesc    = document.getElementById('domain-desc');
const modalOverlay  = document.getElementById('modal-overlay');
const modalContent  = document.getElementById('modal-content');
const modalClose    = document.getElementById('modal-close');

// ── 初始化 ───────────────────────────────────────────────────────────
function init() {
  Object.keys(DOMAINS).forEach(d => {
    sessions[d]  = `session_${d}_${Date.now()}`;
    userIds[d]   = 'user001';
    histories[d] = [];
  });
  switchDomain('medical');
  bindEvents();
}

// ── 绑定事件 ─────────────────────────────────────────────────────────
function bindEvents() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchDomain(btn.dataset.domain));
  });

  sendBtn.addEventListener('click', sendMessage);
  textarea.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  textarea.addEventListener('input', autoResize);

  userIdInput.addEventListener('change', () => {
    userIds[currentDomain] = userIdInput.value.trim() || 'user001';
  });

  clearBtn.addEventListener('click', clearMemories);
  refreshBtn.addEventListener('click', () => loadMemoryPanel(currentDomain, userIds[currentDomain]));

  modalOverlay.addEventListener('click', e => { if (e.target === modalOverlay) closeModal(); });
  modalClose.addEventListener('click', closeModal);
}

// ── 切换 domain ──────────────────────────────────────────────────────
function switchDomain(domain) {
  currentDomain = domain;
  const info = DOMAINS[domain];

  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.domain === domain));
  domainBadge.textContent = info.label;
  domainBadge.className = `domain-badge ${domain}`;
  domainDesc.textContent = info.desc;
  userIdInput.value = userIds[domain];

  renderHistory(domain);
  loadMemoryPanel(domain, userIds[domain]);
}

// ── 渲染历史消息 ─────────────────────────────────────────────────────
function renderHistory(domain) {
  messagesWrap.innerHTML = '';
  histories[domain].forEach(item => appendMessage(item, false));
  scrollBottom();
}

// ── 发送消息 ─────────────────────────────────────────────────────────
async function sendMessage() {
  const query = textarea.value.trim();
  if (!query) return;
  const domain = currentDomain;
  const userId = userIds[domain];

  // 记录并渲染用户消息
  const userMsg = { role: 'user', content: query };
  histories[domain].push(userMsg);
  appendMessage(userMsg);
  textarea.value = '';
  autoResize();
  sendBtn.disabled = true;

  // 打字动画
  const typingEl = showTyping();
  scrollBottom();

  try {
    const res = await apiChat(domain, userId, query, sessions[domain]);
    typingEl.remove();

    lastRecalled = res.recalled || [];
    const botMsg = { role: 'assistant', content: res.reply, recalled: lastRecalled };
    histories[domain].push(botMsg);
    appendMessage(botMsg);

    // 刷新记忆面板（本轮召回高亮）
    await loadMemoryPanel(domain, userId, lastRecalled.map(r => r.memory));
  } catch (err) {
    typingEl.remove();
    const errMsg = { role: 'assistant', content: `❌ 请求失败：${err.message}`, recalled: [] };
    histories[domain].push(errMsg);
    appendMessage(errMsg);
  } finally {
    sendBtn.disabled = false;
    scrollBottom();
  }
}

// ── 渲染单条消息 ─────────────────────────────────────────────────────
function appendMessage(msg, scroll = true) {
  const isUser = msg.role === 'user';
  const row = document.createElement('div');
  row.className = `msg-row ${isUser ? 'user' : 'bot'}`;

  const avatar = document.createElement('div');
  avatar.className = `avatar ${isUser ? 'user' : 'bot'}`;
  avatar.textContent = isUser ? '👤' : DOMAINS[currentDomain].emoji;

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = msg.content;

  // recalled 标签（只有 assistant 且有召回时显示）
  if (!isUser && msg.recalled && msg.recalled.length > 0) {
    const tag = document.createElement('div');
    tag.className = 'recalled-tag';
    tag.textContent = `📎 引用了 ${msg.recalled.length} 条历史记忆`;
    tag.addEventListener('click', () => showRecalledModal(msg.recalled));
    bubble.appendChild(tag);
  }

  const meta = document.createElement('div');
  meta.className = 'bubble-meta';
  meta.textContent = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });

  const col = document.createElement('div');
  col.style.display = 'flex'; col.style.flexDirection = 'column';
  col.appendChild(bubble); col.appendChild(meta);

  row.appendChild(avatar); row.appendChild(col);
  messagesWrap.appendChild(row);
  if (scroll) scrollBottom();
}

// ── 打字动画 ─────────────────────────────────────────────────────────
function showTyping() {
  const row = document.createElement('div');
  row.className = 'msg-row bot';
  row.innerHTML = `
    <div class="avatar bot">${DOMAINS[currentDomain].emoji}</div>
    <div class="bubble typing-indicator">
      <span></span><span></span><span></span>
    </div>`;
  messagesWrap.appendChild(row);
  return row;
}

// ── 记忆面板 ─────────────────────────────────────────────────────────
async function loadMemoryPanel(domain, userId, recalledTexts = []) {
  memoryList.innerHTML = '<div class="memory-empty">加载中…</div>';
  try {
    const data = await apiListMemories(domain, userId);
    renderMemoryPanel(data.memories, recalledTexts);
  } catch {
    memoryList.innerHTML = '<div class="memory-empty">（加载失败）</div>';
  }
}

function renderMemoryPanel(memories, recalledTexts = []) {
  memoryList.innerHTML = '';
  if (!memories || memories.length === 0) {
    memoryList.innerHTML = '<div class="memory-empty">暂无记忆</div>';
    return;
  }
  memories.forEach(m => {
    const card = document.createElement('div');
    const isRecalled = recalledTexts.some(t => t && m.memory && m.memory.includes(t.slice(0, 20)));
    card.className = `memory-card${isRecalled ? ' recalled' : ''}`;
    card.textContent = m.memory;
    memoryList.appendChild(card);
  });
}

// ── 清空记忆 ─────────────────────────────────────────────────────────
async function clearMemories() {
  const domain = currentDomain;
  const userId = userIds[domain];
  if (!confirm(`确认清空「${DOMAINS[domain].label}」中用户「${userId}」的全部记忆？`)) return;
  try {
    const res = await apiClearMemories(domain, userId);
    alert(res.message);
    await loadMemoryPanel(domain, userId);
  } catch (err) {
    alert(`清空失败：${err.message}`);
  }
}

// ── 模态框（recalled 详情）───────────────────────────────────────────
function showRecalledModal(recalled) {
  modalContent.innerHTML = '';
  recalled.forEach((r, i) => {
    const row = document.createElement('div');
    row.className = 'mem-row';
    row.innerHTML = `<strong>${i + 1}.</strong> ${escapeHtml(r.memory)}
      <span class="mem-score">相似度 ${r.score}</span>`;
    modalContent.appendChild(row);
  });
  modalOverlay.classList.add('show');
}
function closeModal() { modalOverlay.classList.remove('show'); }

// ── 工具 ─────────────────────────────────────────────────────────────
function scrollBottom() {
  messagesWrap.scrollTop = messagesWrap.scrollHeight;
}
function autoResize() {
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}
function escapeHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── 启动 ─────────────────────────────────────────────────────────────
init();
