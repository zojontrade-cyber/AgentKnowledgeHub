'use strict';
/* ============================================================
   AgentKnowledgeHub · 索引台 前端逻辑
   设计系统见 docs/ui-design-system.md
   约束：无框架、无构建，单文件原生 JS，直接由 FastAPI /static 提供。
   ============================================================ */

const API = '';
let API_KEY = localStorage.getItem('akh_key') || 'dev-key-1';

/* ── 会话 id：客户端自有 ──────────────────────────────────────
   服务端**不会**自动复用匿名 id（未带 session_id 的调用是匿名单次问答，
   响应里 session_id 为 null）。因此想累计就用这里生成的 id 一直带着。
   轮次则相反：由服务端按数据库记录递增取号，前端不自算权威值。 */
const SESSION_KEY = 'akh_session';
const SESSION_ID_RE = /^[A-Za-z0-9_-]{1,64}$/;

function newSessionId() {
  try {
    if (window.crypto && typeof window.crypto.randomUUID === 'function') {
      return window.crypto.randomUUID().replace(/-/g, '');
    }
  } catch (_) { /* 非安全上下文等 -> 走下面的回退 */ }
  let s = '';
  for (let i = 0; i < 32; i++) s += Math.floor(Math.random() * 16).toString(16);
  return s;
}

function sessionId() {
  let id = null;
  try { id = sessionStorage.getItem(SESSION_KEY); } catch (_) { /* 隐私模式 */ }
  if (!id || !SESSION_ID_RE.test(id)) {
    id = newSessionId();
    try { sessionStorage.setItem(SESSION_KEY, id); } catch (_) { /* 忽略 */ }
  }
  return id;
}

const SESSION_ID = sessionId();

/* 检索通道：与 services/graph_rag.py 的 source_type 一致。
   weight 是 _cross_rerank 里的重排权重——展示它，用户才知道
   "为什么这条排在前面"，而不是把一个被加权过的分数谎称为"相关度"。 */
const CHANNELS = {
  path:       { zh: '推理路径', weight: 1.25, hint: '实体对之间的最短关系路径' },
  vector:     { zh: '向量检索', weight: 1.00, hint: '语义相近的文档块' },
};
const CHANNEL_ORDER = ['vector'];

/* 分类配色 */
const TYPE_COLOR = {
  Person: '#D9A05B', Organization: '#6FA8C8', Event: '#B07CC6',
  Artifact: '#7FBF9B', Location: '#C97F7F', Concept: '#6E7F80',
};
const TYPE_ZH = {
  Person: '人物', Organization: '组织', Event: '事件',
  Artifact: '人造物', Location: '地点', Concept: '常识概念',
};

/* 关系族 → 颜色，复用同一套 6 个色相 */
const REL_FAMILY = {
  Causes: 'R_CAUSE', CausesDesire: 'R_CAUSE',
  CapableOf: 'R_CAP', Desires: 'R_CAP',
  AtLocation: 'R_LOC', LocatedNear: 'R_LOC',
  HasA: 'R_CMP', HasProperty: 'R_CMP', PartOf: 'R_CMP',
  FormOf: 'R_CMP', DerivedFrom: 'R_CMP',
  HasSubevent: 'R_SEQ', HasFirstSubevent: 'R_SEQ',
  RELATED_TO: 'R_REL', Antonym: 'R_REL', DistinctFrom: 'R_REL',
  EtymologicallyRelatedTo: 'R_REL', EtymologicallyDerivedFrom: 'R_REL',
};
const FAMILY_COLOR = {
  R_CAUSE: '#C97F7F', R_CAP: '#D9A05B', R_LOC: '#6FA8C8',
  R_CMP: '#7FBF9B', R_SEQ: '#B07CC6', R_REL: '#6E7F80',
};
const FAMILY_ZH = {
  R_CAUSE: '因果', R_CAP: '能力与意愿', R_LOC: '位置',
  R_CMP: '构成', R_SEQ: '时序', R_REL: '一般关联',
};

function colorOf(group) {
  return TYPE_COLOR[group] || FAMILY_COLOR[group] || TYPE_COLOR.Concept;
}
function zhOf(group) {
  return TYPE_ZH[group] || FAMILY_ZH[group] || group;
}

/* 关系名中文化 */
const RELATION_ZH = {
  RELATED_TO: '相关', BELONGS_TO: '隶属', WORKS_AT: '就职于', LOCATED_IN: '位于',
  DEVELOPED_BY: '开发', PART_OF: '属于', USES: '使用', DEPENDS_ON: '依赖',
  CONTAINS: '包含', APPLIES_TO: '适用于', APPROVED_BY: '审批', REQUIRES: '需要',
  PROHIBITS: '禁止', REGULATES: '规定', DEFINES: '定义', REPORTS_TO: '汇报给',
  COMPETES_WITH: '竞争', SUBSIDIARY_OF: '子公司', RESPONSIBLE_FOR: '负责',
  Causes: '导致', CapableOf: '能够', Desires: '想要', AtLocation: '位于',
  CausesDesire: '引发想要', HasFirstSubevent: '首个子事件', HasSubevent: '子事件',
  HasA: '含有', HasProperty: '属性', DerivedFrom: '衍生自', FormOf: '形式',
  Antonym: '反义', DistinctFrom: '不同于', EtymologicallyRelatedTo: '词源相关',
  EtymologicallyDerivedFrom: '词源衍生',
};

const DOC_STATUS = {
  PENDING:     { zh: '排队中',   cls: 'is-run' },
  PROCESSING:  { zh: '处理中',   cls: 'is-run' },
  VECTOR_DONE: { zh: '向量完成', cls: 'is-run' },
  COMMITTED:   { zh: '已入库',   cls: 'is-ok' },
  FAILED:      { zh: '失败',     cls: 'is-err' },
};

const VIEW_TITLE = { chat: '智能问答', upload: '知识入库', overview: '数据概览' };

/* ── 基础工具 ───────────────────────────────────────────────── */

function toast(msg, kind) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'toast is-show' + (kind ? ' ' + kind : '');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.className = 'toast'; }, 3600);
}

async function api(path, opts) {
  opts = opts || {};
  const headers = opts.headers || {};
  headers['X-API-Key'] = API_KEY;
  if (opts.body && !(opts.body instanceof FormData)) headers['Content-Type'] = 'application/json';
  const res = await fetch(API + path, Object.assign({}, opts, { headers }));
  const text = await res.text();
  let data;
  try { data = JSON.parse(text); } catch (_) { data = { detail: text }; }
  if (!res.ok) {
    const detail = data && data.detail;
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail || data));
  }
  return data;
}

function escapeHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function num(n) {
  return (n == null ? 0 : n).toLocaleString('zh-CN');
}

/* 缓存命中率显示。务必与调用次数比例区分：
   这是「可测 Prompt Token 的命中占比」= Hit / (Hit + Miss)。 */
function fmtPct(v) {
  return (Number(v || 0) * 100).toFixed(1) + '%';
}

function fmtTokens(u) {
  return (u && u.total_tokens) ? num(u.total_tokens) : '—';
}

/* 单次答案的缓存格文案 + tooltip。缺失与"命中 0"必须区分：
   - 不可测（provider 不报字段）-> 「—」并说明原因
   - 可测但 hit=0               -> 「0.0%」（这是真实信息，不是缺失） */
function cacheCell(u) {
  if (!u) return { text: '—', title: '未返回用量信息' };
  const measured = u.cache_measured_calls || 0;
  const hit = u.cache_hit_tokens || 0;
  const miss = u.cache_miss_tokens || 0;
  const calls = u.llm_calls || 0;

  if (hit + miss === 0) {
    return {
      text: '—',
      title: measured
        ? '本次可测调用的 Prompt Token 为 0，命中率无意义'
        : '当前模型 / 接口不返回缓存字段，无法判断命中情况',
    };
  }

  let text = fmtPct(hit / (hit + miss));
  if (!u.cache_supported) text += '（' + measured + '/' + calls + ' 次可测）';
  return {
    text: text,
    title: '按可测 Prompt Token 计算：Hit / (Hit + Miss)。命中 ' + num(hit) +
      ' / 未命中 ' + num(miss) + '。另有 ' + (u.cache_hit_calls || 0) + '/' + measured +
      ' 次调用发生了命中 —— 这是另一个概念，不是上面的命中率。',
  };
}

function isMobile() {
  return window.matchMedia('(max-width:999px)').matches;
}

/* ── 视图切换 ───────────────────────────────────────────────── */

const navButtons = Array.prototype.slice.call(document.querySelectorAll('.rail-nav button'));

function setView(view) {
  navButtons.forEach(b => {
    const on = b.dataset.view === view;
    if (on) b.setAttribute('aria-current', 'true'); else b.removeAttribute('aria-current');
  });
  document.querySelectorAll('.view').forEach(v => v.classList.remove('is-active'));
  const section = document.getElementById('view-' + view);
  if (section) section.classList.add('is-active');
  document.title = (VIEW_TITLE[view] || '索引台') + ' · AgentKnowledgeHub';

  if (view === 'overview') loadOverview();
  if (view === 'upload') loadDocuments();
  if (view === 'chat' && !isMobile()) document.getElementById('question').focus();
}

navButtons.forEach(b => { b.onclick = () => setView(b.dataset.view); });

/* ── 系统状态 + 台账 ───────────────────────────────────────── */

function paintLedger(d) {
  const vs = (d && d.vector_store) || {};
  const set = (id, val) => {
    const el = document.getElementById(id);
    if (!el) return;
    if (val == null) { el.textContent = '—'; el.classList.add('is-empty'); }
    else { el.textContent = num(val); el.classList.remove('is-empty'); }
  };
  set('lg-docs', d ? d.uploaded_files : null);
  set('lg-vectors', d ? vs.total_vectors : null);
}

/* 侧栏「本次会话」：数字全部来自服务端，前端不自算权威值。
   degraded=true 时（统计库写入失败）显示"统计不可用"，
   而不是把 fallback 的 turn=1 当真值展示。 */
function paintSession(s, degraded) {
  const put = (id, text, empty, title) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    el.classList.toggle('is-empty', !!empty);
    if (title !== undefined) el.title = title;
  };

  if (degraded) {
    put('lg-turns', '统计不可用', false, '用量统计库写入失败，本轮数字仅本请求有效');
    return;
  }
  if (!s || !s.turns) {
    put('lg-turns', '—', true);
    put('lg-tokens', '—', true);
    put('lg-cache', '—', true, '按可测 Prompt Token 计算：Hit / (Hit + Miss)');
    return;
  }

  put('lg-turns', num(s.turns) + ' 轮', false);
  put('lg-tokens', num(s.total_tokens), !s.total_tokens);

  const hit = s.cache_hit_tokens || 0;
  const miss = s.cache_miss_tokens || 0;
  if (hit + miss === 0) {
    put('lg-cache', '—', true, '当前模型 / 接口不返回缓存字段，无法判断命中情况');
  } else {
    put('lg-cache', fmtPct(hit / (hit + miss)), false,
      '按可测 Prompt Token 计算：Hit / (Hit + Miss)。累计命中 ' + num(hit) +
      ' / 未命中 ' + num(miss));
  }
}

/* 刷新 / 重开后把侧栏数字读回来 —— 否则"持久化"只是名义上的。
   旧进程没有该路由（404）时静默降级，不弹错。 */
async function restoreSession() {
  try {
    const d = await api('/api/ui/qa-session?session_id=' +
      encodeURIComponent(SESSION_ID) + '&limit=10');
    paintSession(d.summary, false);
  } catch (_) {
    paintSession(null, false);
  }
}

/* 刷新 / 重开后恢复历史问答卡片。

   库里存的是**当轮响应快照**（问题/答案/出处/推理步骤/用量），形状就是
   renderQA/renderProv 需要的 reply，因此这里复用**与实时回答完全相同**的
   渲染路径 —— 不可能出现"恢复出来的卡片和当时看到的不一样"。

   注意：历史**只用于展示**，不会注入 prompt，问答链路仍是单轮无记忆。
   详见 docs/conversation-history.md */
async function restoreHistory() {
  let d;
  try {
    d = await api('/api/ui/qa-history?session_id=' +
      encodeURIComponent(SESSION_ID) + '&limit=50');
  } catch (_) {
    return;   // 旧进程无该路由(404) 或网络失败：静默降级，实时问答不受影响
  }

  const turns = (d && d.turns) || [];
  if (!turns.length) return;

  const welcome = document.getElementById('welcome');
  if (welcome) welcome.remove();

  if (d.truncated) {
    const note = document.createElement('p');
    note.className = 'qa-note';
    note.textContent = '仅显示最近 ' + turns.length + ' 轮（该会话共 ' + d.total + ' 轮）';
    askCol.appendChild(note);
  }

  turns.forEach(t => renderQA(t.question, t));

  // 抽屉语义是"本次检索" = 最后一轮；点历史卡片的引用会切到该轮（见 selectCite）
  const last = turns[turns.length - 1];
  qa.sources = last.sources || [];
  qa.steps = last.reasoning_steps || [];
  qa.channel = null;
  qa.usage = last.usage || null;
  qa.turn = last.turn || null;
  qa.degraded = !!last.metrics_degraded;
  renderProv(false);

  askScroll.scrollTop = askScroll.scrollHeight;
}

/* 依赖状态语义照抄 api/main.py：
   healthy = all(v == "ok" for k, v in deps.items() if k != "reranker")
   —— reranker 只影响排序质量，不可用时会 fail-open 回退 BM25，
   把它当成故障会误报（旧版把 disabled 显示为红色异常）。 */
function depState(deps) {
  const keys = Object.keys(deps || {});
  if (!keys.length) return { level: '', text: '正在检查服务' };
  const critical = keys.filter(k => k !== 'reranker' && deps[k] !== 'ok');
  const rr = deps.reranker;
  if (critical.length) return { level: 'is-bad', text: '异常：' + critical.join('、') };
  if (rr === 'disabled') return { level: '', text: '服务运行正常（重排已关闭）' };
  if (rr && rr !== 'ok') return { level: 'is-warn', text: '服务正常，重排已降级' };
  return { level: '', text: '服务运行正常' };
}

async function checkHealth() {
  const box = document.getElementById('rail-status');
  const label = document.getElementById('rail-status-text');
  try {
    const d = await api('/api/health');
    const st = depState(d.dependencies);
    box.className = 'rail-status' + (st.level ? ' ' + st.level : '');
    label.textContent = st.text;
  } catch (_) {
    box.className = 'rail-status is-bad';
    label.textContent = '服务未连接';
  }
}

/* ── 问答 ───────────────────────────────────────────────────── */

const askScroll = document.getElementById('ask-scroll');
const askCol = document.getElementById('ask-col');
const questionEl = document.getElementById('question');
const askForm = document.getElementById('ask-form');
const askBtn = document.getElementById('ask-btn');
const provBox = document.getElementById('prov');
const provBody = document.getElementById('prov-body');
const provToggle = document.getElementById('prov-toggle');
const provSummary = document.getElementById('prov-summary');

const qa = { sources: [], steps: [], channel: null, qaEl: null, usage: null, turn: null, degraded: false };

document.querySelectorAll('#trials button').forEach(b => {
  b.onclick = () => { questionEl.value = b.dataset.q; ask(); };
});

questionEl.addEventListener('input', () => {
  questionEl.style.height = 'auto';
  questionEl.style.height = Math.min(questionEl.scrollHeight, 160) + 'px';
});
questionEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(); }
});
askForm.addEventListener('submit', e => { e.preventDefault(); ask(); });

provToggle.addEventListener('click', () => {
  const open = provBox.classList.toggle('is-open');
  provToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
});

/* 由 reply 文本构造答案卡。引证不做句级内联：后端只返回按重排得分排序的
   出处集合，没有"哪句对应哪条"的对应关系，凭空插标就是编造溯源。
   所以引证集中放在答案末尾一行，并注明排序依据。 */
function renderQA(question, reply) {
  const wrap = document.createElement('article');
  wrap.className = 'qa';

  const sources = (reply && reply.sources) || [];
  const steps = (reply && reply.reasoning_steps) || [];
  const err = reply && reply.error;

  let cites = '';
  if (sources.length) {
    cites = '<div class="cite-row" role="group" aria-label="本条答案的出处">' +
      '<span class="cite-row-label">出处</span>' +
      sources.map((s, i) =>
        '<button class="cite" type="button" data-idx="' + i + '" aria-current="false" ' +
        'aria-label="出处 ' + (i + 1) + '：' + escapeHtml(channelOf(s).zh) + '">' + (i + 1) + '</button>'
      ).join('') +
      '<span class="cite-note">按重排得分排列，右侧可逐条核验</span>' +
      '<button class="cite-open" type="button">查看 ' + sources.length + ' 条出处</button>' +
      '</div>';
  }

  // 用量：轮次由服务端计号；token / 缓存取自本轮各次 LLM 调用的实测值
  //
  // 一律从 `reply` 读，**不读模块级状态**（qa.degraded / qa.turn）：
  // 恢复历史时会把多条 reply 依次交给本函数，若读模块状态，
  // 所有历史卡片都会被套上"当前这一轮"的轮次与降级标记。
  const u = (reply && reply.usage) || null;
  const cc = cacheCell(u);
  const degraded = !!(reply && reply.metrics_degraded);
  const turnText = degraded ? '—' : (reply && reply.turn ? '第 ' + reply.turn + ' 轮' : '—');
  const tokTitle = u
    ? '输入 ' + num(u.prompt_tokens) + ' / 输出 ' + num(u.completion_tokens) +
      '；共 ' + (u.llm_calls || 0) + ' 次模型调用'
    : '未返回用量信息';
  const latText = (u && u.latency_ms) ? (u.latency_ms / 1000).toFixed(1) + 's' : '—';

  const readout = err ? '' :
    '<dl class="readout">' +
      '<div class="cell"><dt>意图</dt><dd>' + escapeHtml(reply.intent || '—') + '</dd></div>' +
      '<div class="cell"><dt>置信度</dt><dd>' + Math.round((reply.confidence || 0) * 100) + '%</dd></div>' +
      '<div class="cell"><dt>出处</dt><dd>' + sources.length + '</dd></div>' +
      '<div class="cell"><dt>检索步骤</dt><dd>' + steps.length + '</dd></div>' +
      '<div class="cell"><dt>轮次</dt><dd>' + escapeHtml(turnText) + '</dd></div>' +
      '<div class="cell"><dt>Tokens</dt><dd title="' + escapeHtml(tokTitle) + '">' + fmtTokens(u) + '</dd></div>' +
      '<div class="cell"><dt>缓存命中</dt><dd title="' + escapeHtml(cc.title) + '">' + escapeHtml(cc.text) + '</dd></div>' +
      '<div class="cell"><dt>耗时</dt><dd>' + escapeHtml(latText) + '</dd></div>' +
    '</dl>';

  const degradedNote = (degraded && !err)
    ? '<p class="qa-note">用量统计库写入失败，本轮数字仅本请求有效，会话累计未更新。</p>'
    : '';

  const channelBar = (!err && sources.length)
    ? '<div class="channels" role="group" aria-label="按检索通道筛选出处">' + channelChips(sources) + '</div>'
    : '';

  wrap.innerHTML =
    '<div class="q-line"><span class="q-mark" aria-hidden="true">问</span><h2>' + escapeHtml(question) + '</h2></div>' +
    '<div class="answer ' + (err ? 'is-error' : '') + '">' +
      (err ? escapeHtml(err) : escapeHtml(reply.answer || '')) +
    '</div>' +
    degradedNote + cites + readout + channelBar;

  askCol.appendChild(wrap);
  qa.qaEl = wrap;
  // 记住本轮 reply：历史卡片与抽屉之间靠它对应（见 selectCite）
  wrap._reply = reply;

  wrap.querySelectorAll('.cite').forEach(b => {
    b.onclick = () => selectCite(Number(b.dataset.idx), wrap);
  });
  const opener = wrap.querySelector('.cite-open');
  if (opener) opener.onclick = () => openProv();
  wrap.querySelectorAll('.chan').forEach(b => {
    b.onclick = () => setChannel(qa.channel === b.dataset.channel ? null : b.dataset.channel);
  });
}

function openProv() {
  provBox.classList.add('is-open');
  provToggle.setAttribute('aria-expanded', 'true');
}

function channelOf(s) {
  const t = s && s.type;
  if (CHANNELS[t]) return { key: t, zh: CHANNELS[t].zh, weight: CHANNELS[t].weight, known: true };
  return { key: t || 'other', zh: t || '其他来源', weight: null, known: false };
}

function channelChips(sources) {
  const counts = {};
  sources.forEach(s => {
    const c = channelOf(s);
    counts[c.key] = counts[c.key] || { n: 0, zh: c.zh, weight: c.weight };
    counts[c.key].n += 1;
  });
  const keys = Object.keys(counts).sort((a, b) => {
    const ia = CHANNEL_ORDER.indexOf(a), ib = CHANNEL_ORDER.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
  return keys.map(k => {
    const c = counts[k];
    const w = c.weight == null ? '' : ' ×' + c.weight.toFixed(2);
    return '<button class="chan" type="button" data-channel="' + escapeHtml(k) + '" aria-pressed="' +
      (qa.channel === k ? 'true' : 'false') + '">' + escapeHtml(c.zh) + w +
      ' <b>' + c.n + '</b></button>';
  }).join('');
}

function setChannel(key) {
  qa.channel = key;
  if (qa.qaEl) {
    qa.qaEl.querySelectorAll('.chan').forEach(b => {
      b.setAttribute('aria-pressed', b.dataset.channel === key ? 'true' : 'false');
    });
  }
  renderProv(false);
}

/* 出处条目的元信息：按通道取真实字段，不编造 */
function sourceMeta(s, ch) {
  const m = (s && s.metadata) || {};
  if (ch.key === 'vector') {
    const bits = [];
    if (m.source) bits.push(String(m.source));
    if (m.heading_path) bits.push(String(m.heading_path));
    else if (m.section_title) bits.push(String(m.section_title));
    if (m.chunk_index != null && m.chunk_index !== '') bits.push('块 #' + m.chunk_index);
    return bits.join(' · ');
  }
  return '';
}

/* 用量拆解：把混合的总命中率拆成
     ① Prompt Cache Hit Rate（可测 Prompt Token 的命中占比）
     ② 稳定前缀复用率（system 段 —— 所有问题逐字节相同的那部分）
     ③ 检索上下文复用率（RAG 上下文）
   只显示后端给得出的项；取不到就写「—」或说明原因，不猜。 */
function usageBreakdown(u) {
  if (!u) return '';

  const pctOr = (v) => (v == null ? '—' : fmtPct(v));
  const rows = [
    ['Prompt tokens', num(u.prompt_tokens)],
    ['① 命中率 Hit/(Hit+Miss)',
      pctOr(u.cache_hit_rate) + '（命中 ' + num(u.cache_hit_tokens) +
      ' / 未命中 ' + num(u.cache_miss_tokens) + '）'],
    ['② 稳定前缀复用', pctOr(u.stable_prefix_reuse_rate) +
      '（system 段约 ' + num(u.stable_prefix_tokens) + ' token）'],
    ['③ 检索上下文复用', pctOr(u.rag_context_reuse_rate) +
      '（上下文约 ' + num(u.context_tokens) + ' token）'],
    ['用户问题 tokens', num(u.query_tokens)],
    ['Completion tokens', num(u.completion_tokens)],
  ];

  // 缓存写入侧：本 Provider 不给数值 —— 必须写清"不可用"，不能显示成 0
  rows.push(['缓存写入 tokens',
    u.cache_creation_tokens == null
      ? '<span class="dim">Provider 未提供</span>'
      : num(u.cache_creation_tokens)]);

  return '<div class="breakdown">' +
    '<h3>用量拆解</h3>' +
    '<dl class="breakdown-list">' +
      rows.map(r => '<div class="breakdown-row"><dt>' + r[0] + '</dt><dd>' + r[1] + '</dd></div>').join('') +
    '</dl>' +
    '<p class="calls-note">②③ 是按字符比例摊分的**估算**（精确值是 Prompt tokens）；' +
    '命中量优先归给稳定前缀，剩余才算上下文复用，属保守归因。</p>' +
  '</div>';
}

/* 按 stage 展开本轮每一次 LLM 调用。
   这是"token 为什么这么高"的唯一可定位视图 —— 轮级汇总答不了这个。 */
function callTable(u) {
  const calls = (u && u.per_call) || [];
  if (!calls.length) return '';

  const rows = calls.map(c => {
    let cache;
    if (!c.cache_supported) {
      cache = '<span class="dim">不可测</span>';
    } else {
      cache = num(c.cache_hit_tokens) + ' / ' + num(c.cache_miss_tokens);
      // ● 只标注"确实命中"，与不可测、命中 0 区分开
      if (c.cache_hit_tokens > 0) cache += ' <b class="dot" title="该次调用命中缓存">●</b>';
    }
    return '<tr>' +
      '<td class="mono">' + escapeHtml(String(c.call_index || '')) + '</td>' +
      '<td>' + escapeHtml(c.stage || '—') + '</td>' +
      '<td class="mono">' + num(c.prompt_tokens) + '</td>' +
      '<td class="mono">' + num(c.completion_tokens) + '</td>' +
      '<td class="mono">' + cache + '</td>' +
    '</tr>';
  }).join('');

  const measured = u.cache_measured_calls || 0;
  const note = (measured < (u.llm_calls || 0))
    ? '<p class="calls-note">' + measured + '/' + (u.llm_calls || 0) +
      ' 次调用给出了可识别的缓存字段；命中率只在可测调用内计算。</p>'
    : '<p class="calls-note">命中率按可测 Prompt Token 计算（Hit / (Hit + Miss)），' +
      '不是"多少次调用命中"。</p>';

  return '<div class="calls">' +
    '<h3>模型调用（' + (u.llm_calls || 0) + ' 次）</h3>' +
    '<table class="call-table"><thead><tr>' +
      '<th scope="col">#</th><th scope="col">阶段</th>' +
      '<th scope="col">输入</th><th scope="col">输出</th>' +
      '<th scope="col">缓存 命中/未命中</th>' +
    '</tr></thead><tbody>' + rows + '</tbody></table>' + note +
  '</div>';
}

function renderProv(animate) {
  const all = qa.sources;
  const list = qa.channel ? all.filter(s => channelOf(s).key === qa.channel) : all;

  const nEl = document.getElementById('prov-toggle-n');
  if (nEl) nEl.textContent = String(all.length);

  document.getElementById('prov-summary').textContent = all.length
    ? (new Set(all.map(s => channelOf(s).key)).size + ' 个通道 · ' + all.length + ' 条出处' +
       (qa.channel ? '（筛选中）' : ''))
    : '尚未提问';

  if (!all.length) {
    provBody.innerHTML = '<p class="prov-empty">提问后，这里会按检索通道列出每一条出处：' +
      '文件名、章节位置、块号，以及该通道在重排时的权重。</p>';
    return;
  }

  let html = '';
  if (qa.channel) {
    const c = CHANNELS[qa.channel];
    html += '<p class="prov-group">' + escapeHtml(c ? c.zh : qa.channel) +
      (c ? '（权重 ×' + c.weight.toFixed(2) + '）' : '') + '</p>';
  }
  list.forEach((s) => {
    const idx = all.indexOf(s);
    const ch = channelOf(s);
    const meta = sourceMeta(s, ch);
    html += '<button class="prov-item' + (animate ? ' is-new' : '') + '" type="button" data-idx="' + idx +
      '" aria-current="false"' + (animate ? ' style="--i:' + Math.min(idx, 12) + '"' : '') + '>' +
      '<span class="prov-idx">' + (idx + 1) + '</span>' +
      '<span class="prov-main">' +
        '<span class="prov-kind">' + escapeHtml(ch.zh) +
          (ch.weight == null ? '' : ' <em>×' + ch.weight.toFixed(2) + '</em>') +
          (s.score == null ? '' : ' <span class="prov-w">得分 ' + Number(s.score).toFixed(2) + '</span>') +
        '</span>' +
        '<span class="prov-text">' + escapeHtml(s.content || '') + '</span>' +
        (meta ? '<span class="prov-meta">' + escapeHtml(meta) + '</span>' : '') +
      '</span></button>';
  });

  if (qa.steps.length) {
    html += '<div class="steps"><h3>检索过程</h3><ol>' +
      qa.steps.map(s => '<li>' + escapeHtml(s) + '</li>').join('') + '</ol></div>';
  }

  html += callTable(qa.usage);
  html += usageBreakdown(qa.usage);

  provBody.innerHTML = html;
  provBody.querySelectorAll('.prov-item').forEach(b => {
    b.onclick = () => selectCite(Number(b.dataset.idx));
  });
}

/* 选中某条引用。
   `card` 是被点击的答案卡片 —— 恢复历史后会有多张卡片，
   点历史卡片的引用时，抽屉必须切到**那一轮**的出处，
   否则会出现"卡片是第 1 轮、抽屉列的是第 3 轮证据"的错配。 */
function selectCite(idx, card) {
  const target = card || qa.qaEl;

  if (target && target !== qa.qaEl && target._reply) {
    const r = target._reply;
    qa.sources = r.sources || [];
    qa.steps = r.reasoning_steps || [];
    qa.channel = null;
    qa.usage = r.usage || null;
    qa.turn = r.turn || null;
    qa.degraded = !!r.metrics_degraded;
    qa.qaEl = target;
    renderProv(false);
  }

  if (target) {
    target.querySelectorAll('.cite').forEach(b => {
      b.setAttribute('aria-current', Number(b.dataset.idx) === idx ? 'true' : 'false');
    });
  }
  provBody.querySelectorAll('.prov-item').forEach(b => {
    const on = Number(b.dataset.idx) === idx;
    b.setAttribute('aria-current', on ? 'true' : 'false');
    if (on) {
      b.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      if (isMobile()) openProv();
    }
  });
}

async function ask() {
  const q = questionEl.value.trim();
  if (!q) return;
  questionEl.value = '';
  questionEl.style.height = 'auto';
  askBtn.disabled = true;

  const welcome = document.getElementById('welcome');
  if (welcome) welcome.remove();

  const pending = document.createElement('article');
  pending.className = 'qa';
  pending.innerHTML =
    '<div class="q-line"><span class="q-mark" aria-hidden="true">问</span><h2>' + escapeHtml(q) + '</h2></div>' +
    '<div class="answer is-pending" role="status"><span class="loader"></span>正在检索企业知识库…</div>';
  askCol.appendChild(pending);
  askScroll.scrollTop = askScroll.scrollHeight;

  let reply;
  try {
    // session_id 由客户端自有；轮次不带（服务端计号）。
    reply = await api('/api/qa/ask', {
      method: 'POST',
      body: JSON.stringify({ question: q, session_id: SESSION_ID }),
    });
  } catch (e) {
    pending.remove();
    reply = { error: '检索失败：' + e.message };
    qa.sources = []; qa.steps = []; qa.channel = null;
    qa.usage = null; qa.turn = null; qa.degraded = false;
    renderQA(q, reply);
    renderProv(false);
    toast('提问失败：' + e.message, 'is-err');
    askBtn.disabled = false;
    questionEl.focus();
    askScroll.scrollTop = askScroll.scrollHeight;
    return;
  }

  pending.remove();
  qa.sources = reply.sources || [];
  qa.steps = reply.reasoning_steps || [];
  qa.channel = null;
  // 用量必须在 renderQA 前落位：readout 与「模型调用」表都读它
  qa.usage = reply.usage || null;
  qa.turn = reply.turn || null;
  qa.degraded = !!reply.metrics_degraded;
  renderQA(q, reply);
  renderProv(true);
  // 累计值以服务端返回为准（含本次），避免前端自算与服务端计号漂移
  paintSession(reply.session, qa.degraded);

  // 不做自动展开：手机上抽屉会立刻盖住刚生成的答案，
  // 改用引证行里的「查看 N 条出处」作为显式入口。

  askBtn.disabled = false;
  questionEl.focus();
  askScroll.scrollTop = askScroll.scrollHeight;
}

/* ── 知识入库 ───────────────────────────────────────────────── */

const drop = document.getElementById('drop');
const fileInput = document.getElementById('file-input');
const ingest = document.getElementById('ingest');

drop.onclick = () => fileInput.click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add('is-over'); };
drop.ondragleave = () => drop.classList.remove('is-over');
drop.ondrop = e => {
  e.preventDefault();
  drop.classList.remove('is-over');
  uploadFiles(Array.prototype.slice.call(e.dataTransfer.files));
};
fileInput.onchange = () => uploadFiles(Array.prototype.slice.call(fileInput.files));

const JOB_STEP = {
  PENDING: '已接收，等待处理',
  PROCESSING: '正在解析文档并抽取知识',
  VECTOR_DONE: '向量索引完成',
  GRAPH_DONE: '正在提交',
  COMMITTED: '已入库',
  FAILED: '处理失败',
};

async function uploadFiles(files) {
  if (!files.length) return;
  for (const f of files) {
    const li = document.createElement('li');
    li.innerHTML =
      '<div class="ingest-top"><b>' + escapeHtml(f.name) + '</b><span class="pct">0%</span></div>' +
      '<div class="track"><i></i></div>' +
      '<p class="ingest-step" role="status">正在上传…</p>' +
      '<div class="ingest-counts" hidden></div>';
    ingest.appendChild(li);

    const track = li.querySelector('.track');
    const fill = li.querySelector('.track i');
    const pct = li.querySelector('.pct');
    const step = li.querySelector('.ingest-step');
    const counts = li.querySelector('.ingest-counts');

    const paint = (p, text, cls) => {
      fill.style.transform = 'scaleX(' + Math.max(0, Math.min(100, p)) / 100 + ')';
      pct.textContent = Math.round(p) + '%';
      step.textContent = text;
      step.className = 'ingest-step' + (cls ? ' ' + cls : '');
      if (p >= 100) track.classList.add(cls === 'is-err' ? 'is-err' : 'is-done');
    };

    let docId = null;
    try {
      const fd = new FormData();
      fd.append('file', f);
      const r = await api('/api/ingest/upload', { method: 'POST', body: fd });
      docId = r.doc_id;
      paint(5, (r.message || '已接收，正在后台处理'), '');
    } catch (e) {
      paint(100, '上传失败：' + e.message, 'is-err');
      toast('上传失败：' + e.message, 'is-err');
      continue;
    }

    // 上传是 202 立即返回，真实进度只能从任务接口轮询（旧前端直接读
    // r.chunks_count 一类字段，因此界面上一直显示 undefined）。
    let done = false;
    for (let i = 0; i < 160 && !done; i++) {
      await new Promise(res => setTimeout(res, 1500));
      let job;
      try {
        job = await api('/api/jobs/' + encodeURIComponent(docId));
      } catch (e) {
        paint(5, '进度查询失败：' + e.message, 'is-err');
        break;
      }
      const st = job.status || '';
      paint(job.progress == null ? 0 : job.progress, JOB_STEP[st] || st, st === 'FAILED' ? 'is-err' : '');
      if (st === 'COMMITTED' || st === 'FAILED') {
        done = true;
        if (st === 'FAILED') {
          paint(100, '处理失败：' + (job.error || '未返回错误详情'), 'is-err');
          toast('入库失败：' + f.name, 'is-err');
        } else {
          counts.hidden = false;
          counts.innerHTML =
            '<span>文档块 <b>' + num(job.chunks_count) + '</b></span>' +
            '<span>实体 <b>' + num(job.entities_count) + '</b></span>' +
            '<span>关系 <b>' + num(job.relations_count) + '</b></span>';
          toast('已入库：' + f.name, 'is-ok');
        }
      }
    }
    if (!done) paint(5, '仍在后台处理，可稍后刷新列表查看', '');
  }
  fileInput.value = '';
  loadDocuments();
  loadOverview();
}

async function loadDocuments() {
  const list = document.getElementById('doc-list');
  const cnt = document.getElementById('doc-count');
  try {
    const d = await api('/api/ui/documents');
    cnt.textContent = num(d.total) + ' 份';
    const lg = document.getElementById('lg-docs');
    if (lg) { lg.textContent = num(d.total); lg.classList.remove('is-empty'); }

    if (!d.total) {
      list.innerHTML = '<p class="note">还没有文档。上传一份企业资料，解析、抽取与建索引会在后台自动完成。</p>';
      return;
    }
    list.innerHTML =
      '<table class="table"><thead><tr>' +
        '<th scope="col">文档名称</th><th scope="col">状态</th>' +
        '<th scope="col">大小</th><th scope="col">更新时间</th>' +
      '</tr></thead><tbody>' +
      d.documents.map(x => {
        const st = DOC_STATUS[x.status] || { zh: x.status || '未知', cls: '' };
        return '<tr>' +
          '<td class="name">' + escapeHtml(x.display_name || x.name || '') + '</td>' +
          '<td><span class="state ' + st.cls + '"><i aria-hidden="true"></i>' + escapeHtml(st.zh) + '</span></td>' +
          '<td><span class="num">' + escapeHtml(x.size_human || '') + '</span></td>' +
          '<td><span class="num">' + escapeHtml(new Date((x.modified || 0) * 1000).toLocaleString('zh-CN')) + '</span></td>' +
        '</tr>';
      }).join('') +
      '</tbody></table>';
  } catch (e) {
    list.innerHTML = '<p class="note">文档列表加载失败：' + escapeHtml(e.message) + '</p>';
  }
}

/* ── 数据概览 ───────────────────────────────────────────────── */

const CONFIG_ZH = {
  chat_model: '对话模型', embedding_model: '向量模型',
  chroma_mode: '向量库模式', environment: '运行环境',
};

async function loadOverview() {
  const book = document.getElementById('book');
  const cfg = document.getElementById('config');
  const sig = document.getElementById('signals');
  try {
    const d = await api('/api/ui/overview');
    const vs = d.vector_store || {};
    const put = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = num(v); };

    put('ov-vectors', vs.total_vectors);
    put('ov-files', d.uploaded_files);
    paintLedger(d);

    const c = d.config || {};
    cfg.innerHTML = Object.keys(CONFIG_ZH).map(k =>
      '<dt>' + CONFIG_ZH[k] + '</dt><dd>' + escapeHtml(c[k] || '—') + '</dd>'
    ).join('');

    let deps = {};
    try { deps = (await api('/api/health')).dependencies || {}; } catch (_) { deps = {}; }
    const DEP_ZH = { vector_store: '向量检索服务', reranker: '重排服务' };
    const DEP_VAL = { ok: '正常', disabled: '已关闭（回退 BM25 原序）' };
    const rows = Object.keys(deps).length
      ? Object.keys(deps).map(k => {
          const v = deps[k];
          let cls = 'is-ok';
          if (v !== 'ok') cls = k === 'reranker' ? 'is-warn' : 'is-err';
          return '<div class="state ' + cls + '"><i aria-hidden="true"></i>' +
            escapeHtml(DEP_ZH[k] || k) + '：' +
            escapeHtml(DEP_VAL[v] || String(v)) + '</div>';
        })
      : ['<div class="state"><i aria-hidden="true"></i>未返回依赖状态</div>'];
    rows.push('<div class="state is-ok"><i aria-hidden="true"></i>最近检查 ' +
      escapeHtml(new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })) + '</div>');
    sig.innerHTML = rows.join('');
  } catch (e) {
    const put = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    put('ov-vectors', '—'); put('ov-files', '—');
    cfg.innerHTML = '';
    sig.innerHTML = '<div class="state is-err"><i aria-hidden="true"></i>' + escapeHtml(e.message) + '</div>';
  }
}


/* ── 启动 ─────────────────────────────────────────────────────
   此前 app.js 没有任何引导代码：checkHealth() / paintLedger() 定义了却
   从未被调用，且没有任何视图被标为 is-active —— 而 CSS 里
   `.view{display:none}`，于是首屏是空的（需点一次导航才出现）。
   会话用量回填也必须挂在启动流程里，故在此补齐。 */
(function boot() {
  setView('chat');
  checkHealth();
  loadOverview();      // 内含 paintLedger()
  restoreSession();    // 侧栏「本次会话」回填
  restoreHistory();    // 历史问答卡片恢复（先铺卡片，再让数字与卡片一致）
})();
