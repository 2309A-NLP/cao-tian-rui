// API 基础地址（与后端同源时 '' 即可；跨端口调试时改为 http://localhost:8022）
const API_BASE = '';

/**
 * 发送聊天请求
 * @param {string} domain   - medical | travel | education
 * @param {string} userId   - 用户ID
 * @param {string} query    - 用户输入
 * @param {string|null} sessionId
 * @returns {Promise<{reply:string, recalled:Array}>}
 */
async function apiChat(domain, userId, query, sessionId = null) {
  const resp = await fetch(`${API_BASE}/api/chat/${domain}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId, query, session_id: sessionId }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}

/**
 * 获取用户全部记忆
 * @returns {Promise<{memories:Array, total:number}>}
 */
async function apiListMemories(domain, userId) {
  const resp = await fetch(`${API_BASE}/api/memory/${domain}/${encodeURIComponent(userId)}`);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}

/**
 * 清空用户记忆
 * @returns {Promise<{deleted:number, message:string}>}
 */
async function apiClearMemories(domain, userId) {
  const resp = await fetch(
    `${API_BASE}/api/memory/${domain}/${encodeURIComponent(userId)}`,
    { method: 'DELETE' }
  );
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}
