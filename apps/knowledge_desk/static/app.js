const $ = id => document.getElementById(id);
const state = {session: null, run: null, stream: null, documents: [], tickets: [], lastQuestion: '', view: 'chat', offline: false};
const titles = {chat: '智能对话', library: '知识库', tickets: '工单中心', runs: '运行记录'};
const statuses = {queued: '排队中', running: '处理中', awaiting_approval: '待审核', completed: '已完成', cancelled: '已取消', error: '可恢复'};
const priorities = {low: '低优先级', normal: '普通优先级', high: '高优先级'};
let toastTimer;
let previewQuestion = '';

function node(tag, text, cls) { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (cls) el.className = cls; return el; }
function toast(text) { $('toast').textContent = text; $('toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').hidden = true, 4500); }
async function api(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '输入格式不正确，请检查后重试。');
  return data;
}
const post = (url, body) => api(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});

function showView(view) {
  state.view = view;
  for (const name of Object.keys(titles)) $('view-' + name).hidden = name !== view;
  $('page-title').textContent = titles[view];
  $('inspector').hidden = view !== 'chat';
  document.querySelectorAll('[data-view]').forEach(el => el.classList.toggle('active', el.dataset.view === view));
  if (view === 'library') renderDocuments().catch(error => toast(error.message));
  if (view === 'tickets') refreshTickets().catch(error => toast(error.message));
  if (view === 'runs') renderRuns().catch(error => toast(error.message));
}

async function refresh() {
  const data = await api('/api/bootstrap');
  state.documents = data.documents; state.tickets = data.tickets; state.offline = data.offline;
  state.retrievalMode = data.retrieval_mode;
  if (!previewQuestion) $('search-summary').textContent = '同时查看原始关键词匹配和当前检索的前三条来源。' + (data.retrieval_mode === 'keyword' ? '当前使用关键词与同义词匹配。' : '首次建立语义索引；千问重排会另计用量，相同问题与资料可复用缓存。');
  $('model-name').textContent = data.offline ? '离线示例模式' : data.model;
  $('offline-banner').hidden = !data.offline;
  $('doc-count').textContent = data.documents.length;
  $('ticket-count').textContent = data.tickets.length;
  const list = $('session-list'); list.replaceChildren();
  for (const session of data.sessions) {
    const button = node('button', session.title, 'session-button' + (state.session?.id === session.id ? ' active' : ''));
    button.title = session.title;
    button.addEventListener('click', () => loadSession(session.id).catch(error => toast(error.message)));
    list.append(button);
  }
  return data;
}

function citations(sources) {
  const list = node('div', undefined, 'citations');
  for (const source of sources || []) {
    const button = node('button', `▤ ${source.name.replace(/\.md$/, '')} · ${source.heading}`, 'citation');
    button.addEventListener('click', async () => {
      try {
        const original = await api('/api/sources/' + encodeURIComponent(source.id));
        $('source-name').textContent = original.name;
        $('source-heading').textContent = original.heading + (original.page ? ` · 第 ${original.page} 页` : '');
        $('source-text').textContent = original.text;
        $('source-dialog').showModal();
      } catch (error) { toast(error.message); }
    });
    list.append(button);
  }
  return list;
}

function renderMessages(messages) {
  const list = $('messages'); list.replaceChildren();
  $('welcome').hidden = messages.length > 0;
  for (const message of messages) {
    const row = node('article', undefined, 'message ' + message.role);
    if (message.role === 'assistant') row.append(node('span', '知', 'message-avatar'));
    const body = node('div', undefined, 'message-body');
    if (message.role === 'assistant') body.append(node('div', state.offline ? '知行 · 离线示例' : '知行 · 千问助手', 'message-label'));
    body.append(node('div', message.content));
    if (message.citations?.length) body.append(citations(message.citations));
    const quotes = (message.citations || []).flatMap(source => (source.quotes || []).map(quote => ({source, quote})));
    if (quotes.length) {
      const detail = node('details', undefined, 'answer-evidence');
      detail.append(node('summary', `查看原文依据（${quotes.length} 条摘录）`));
      for (const {source, quote} of quotes) { detail.append(node('blockquote', quote), node('small', source.name + ' · ' + source.heading)); }
      body.append(detail);
    }
    if (message.role === 'user') state.lastQuestion = message.content;
    row.append(body); list.append(row);
  }
}

function updateControls() {
  const busy = state.run && ['queued', 'running', 'awaiting_approval', 'error'].includes(state.run.status);
  $('send').disabled = Boolean(busy);
  $('question').disabled = Boolean(busy);
  $('run-error').hidden = state.run?.status !== 'error';
  $('run-error-text').textContent = state.run?.error || '任务暂停，可从检查点继续。';
}

function renderApproval(draft) {
  $('approval').hidden = !draft;
  if (!draft) return;
  $('draft-title').value = draft.title;
  $('draft-description').value = draft.description;
  $('draft-priority').value = draft.priority;
  $('draft-category').value = draft.category;
  $('approve').disabled = false; $('reject').disabled = false;
}

async function loadSession(id, connect = true) {
  if (state.stream) { state.stream.close(); state.stream = null; }
  showView('chat');
  const snapshot = await api('/api/sessions/' + id);
  state.session = snapshot.session; state.run = snapshot.run;
  renderMessages(snapshot.messages); renderApproval(snapshot.pending); updateControls();
  if (connect) {
    $('timeline').replaceChildren(); $('trace-empty').hidden = Boolean(snapshot.run);
    if (snapshot.run) watchRun(snapshot.run.id);
  }
  await refresh();
}

function renderEvent(event) {
  const data = event.payload;
  if (!['node', 'usage', 'approval', 'error'].includes(event.kind)) return;
  const entry = node('div', undefined, 'trace-entry');
  let text = data.message || (event.kind === 'usage' ? data.purpose + '完成' : '等待你的审核');
  entry.append(node('strong', text));
  let detail = '';
  if (event.kind === 'usage') detail = `${data.model || ''} · ${data.tokens} tokens · ${(data.elapsed_ms / 1000).toFixed(1)} 秒`;
  if (data.sources) detail = data.sources.map(source => source.name + ' / ' + source.heading).join('\n');
  if (detail) entry.append(node('small', detail));
  $('timeline').append(entry); $('trace-empty').hidden = true;
}

function watchRun(id, after = 0) {
  if (state.stream) state.stream.close();
  $('timeline').replaceChildren();
  const stream = new EventSource(`/api/runs/${id}/events?after=${after}`); state.stream = stream;
  let cursor = after;
  stream.addEventListener('trace', event => {
    if (state.stream !== stream) return;
    const item = JSON.parse(event.data);
    if (item.id <= cursor) return;
    cursor = item.id;
    renderEvent(item);
  });
  stream.addEventListener('done', async event => {
    if (state.stream !== stream) return;
    stream.close(); if (state.stream === stream) state.stream = null;
    state.run = JSON.parse(event.data);
    try { await loadSession(state.session.id, false); } catch (error) { toast(error.message); }
  });
  stream.onerror = () => { if (state.stream === stream) toast('进度连接暂时中断，正在重连；任务仍在后台执行。'); };
}

async function send(question, mode = 'auto') {
  if (!question.trim()) return;
  if (state.run && ['queued', 'running', 'awaiting_approval', 'error'].includes(state.run.status)) { toast('请先完成当前任务，或新建对话。'); return; }
  $('send').disabled = true;
  try {
    if (!state.session) state.session = await post('/api/sessions', {});
    state.run = await post(`/api/sessions/${state.session.id}/messages`, {question: question.trim(), mode, request_id: crypto.randomUUID()});
    $('question').value = ''; $('welcome').hidden = true;
    const row = node('article', undefined, 'message user'); row.append(node('div', question, 'message-body')); $('messages').append(row);
    state.lastQuestion = question; $('timeline').replaceChildren(); $('trace-empty').hidden = true;
    updateControls(); watchRun(state.run.id); await refresh();
  } catch (error) { toast(error.message); updateControls(); }
}

async function decide(action) {
  if (!state.run) return;
  const draft = action === 'approve' ? {title: $('draft-title').value, description: $('draft-description').value, priority: $('draft-priority').value, category: $('draft-category').value} : null;
  $('approve').disabled = true; $('reject').disabled = true;
  try {
    state.run = await post(`/api/runs/${state.run.id}/decision`, {action, draft});
    $('approval').hidden = true; updateControls();
    const last = await api('/api/runs/' + state.run.id);
    state.run = last;
    watchRun(last.id);
  } catch (error) { toast(error.message); $('approve').disabled = false; $('reject').disabled = false; }
}

function date(value) { return new Date(value).toLocaleString('zh-CN', {month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit'}); }
async function renderDocuments() {
  state.documents = await api('/api/documents'); $('doc-count').textContent = state.documents.length;
  const list = $('document-list'); list.replaceChildren();
  for (const doc of state.documents) {
    const row = node('tr');
    for (const value of [doc.name, doc.kind.toUpperCase(), doc.chunk_count, date(doc.created_at)]) row.append(node('td', value));
    const action = node('td'); const button = node('button', '移除', 'text-button');
    button.addEventListener('click', async () => { if (!confirm(`移除“${doc.name}”？历史引用将无法再打开。`)) return; try { await api('/api/documents/' + doc.id, {method: 'DELETE'}); await renderDocuments(); } catch (error) { toast(error.message); } });
    action.append(button); row.append(action); list.append(row);
  }
  if (!state.documents.length) { const row = node('tr'); const cell = node('td', '还没有文档，上传后开始提问。', 'empty-state'); cell.colSpan = 5; row.append(cell); list.append(row); }
}

async function upload(files) {
  for (const file of files) {
    if (file.size > 5 * 1024 * 1024) { toast(file.name + ' 超过 5 MB。'); continue; }
    const form = new FormData(); form.append('file', file);
    try { const result = await api('/api/documents', {method: 'POST', body: form}); toast(result.created ? file.name + ' 已加入知识库。' : '文档内容已存在，已避免重复导入。'); } catch (error) { toast(error.message); }
  }
  await renderDocuments(); $('upload-input').value = '';
}

function renderSearchSources(target, sources) {
  const list = $(target); list.replaceChildren();
  if (!sources.length) { list.append(node('p', '没有找到匹配的资料。', 'search-empty')); return; }
  sources.forEach((source, index) => {
    const card = node('article', undefined, 'search-source');
    card.append(node('h3', `${index + 1}. ${source.heading}`), node('small', source.name));
    const channels = source.channels || ['keyword'];
    card.append(node('span', channels.length > 1 ? '关键词与语义均匹配' : channels[0] === 'semantic' ? '语义匹配' : '关键词匹配', 'search-channel'));
    if (source.rerank_grade) card.append(node('span', source.rerank_grade === 3 ? '重排：直接支持' : '重排：部分支持', 'search-channel'));
    card.append(node('p', source.text.slice(0, 240) + (source.text.length > 240 ? '…' : '')));
    card.append(citations([source])); list.append(card);
  });
}

async function previewSearch() {
  const query = $('search-query').value.trim();
  if (!query) { toast('请输入检索问题。'); return; }
  $('search-submit').disabled = true; $('search-query').disabled = true;
  $('ask-search').hidden = true; $('search-comparison').hidden = true;
  $('search-summary').textContent = '正在查找资料…' + (state.retrievalMode === 'keyword' ? '' : '首次检索需要建立语义索引；千问正在核对候选是否包含答案。');
  try {
    const result = await post('/api/search', {query});
    previewQuestion = query;
    $('search-current-title').textContent = ({hybrid: '关键词＋语义匹配', rerank: '混合召回＋千问重排', semantic: '语义匹配', keyword: '同义词增强的关键词匹配'})[result.mode];
    renderSearchSources('search-baseline', result.baseline);
    renderSearchSources('search-current', result.sources);
    $('search-summary').textContent = `${result.fallback ? result.fallback + ' ' : ''}找到 ${result.sources.length} 条来源 · ${(result.elapsed_ms / 1000).toFixed(2)} 秒 · 向量 ${result.embedding_tokens} tokens · 重排 ${result.rerank_tokens || 0} tokens${result.rerank_cache_hits ? '（重排缓存命中）' : ''}。点击引用查看原文，核对是否回答了你的问题。`;
    $('search-comparison').hidden = false; $('ask-search').hidden = false;
  } catch (error) { $('search-summary').textContent = error.message; }
  finally { $('search-submit').disabled = false; $('search-query').disabled = false; }
}

async function refreshTickets() {
  state.tickets = await api('/api/tickets'); $('ticket-count').textContent = state.tickets.length;
  const list = $('ticket-list'); list.replaceChildren();
  for (const ticket of state.tickets) {
    const card = node('article', undefined, 'ticket-card'); const head = node('div', undefined, 'ticket-head'); const title = node('div');
    title.append(node('span', ticket.id, 'ticket-number'), node('h2', ticket.title));
    const select = node('select'); select.setAttribute('aria-label', '工单 ' + ticket.id + ' 的状态');
    for (const [value, label] of [['open', '待处理'], ['in_progress', '处理中'], ['resolved', '已解决']]) { const option = node('option', label); option.value = value; select.append(option); }
    select.value = ticket.status;
    select.addEventListener('change', async () => { try { await api('/api/tickets/' + ticket.id, {method: 'PATCH', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({status: select.value})}); toast('工单状态已更新。'); } catch (error) { select.value = ticket.status; toast(error.message); } });
    head.append(title, select); card.append(head, node('p', ticket.description));
    const meta = node('div', undefined, 'ticket-meta'); meta.append(node('span', priorities[ticket.priority], 'pill'), node('span', ticket.category), node('span', date(ticket.created_at))); card.append(meta); list.append(card);
  }
  if (!state.tickets.length) list.append(node('div', '还没有工单。在对话中描述问题，审核草稿后即可创建。', 'empty-state'));
}

async function renderRuns() {
  const runs = await api('/api/runs'); const list = $('run-list'); list.replaceChildren();
  const stats = $('run-stats'); stats.replaceChildren();
  for (const [label, value] of [['最近任务', runs.length], ['完成任务', runs.filter(r => ['completed', 'cancelled'].includes(r.status)).length], ['模型用量', runs.reduce((n, r) => n + r.tokens, 0).toLocaleString()]]) { const card = node('div', undefined, 'stat-card'); card.append(node('span', label), node('strong', value)); stats.append(card); }
  for (const run of runs) {
    const row = node('tr'); row.append(node('td', run.question.slice(0, 45)));
    const status = node('td'); status.append(node('span', statuses[run.status], 'pill ' + (run.status === 'error' ? 'error' : run.status === 'awaiting_approval' ? 'warning' : 'success'))); row.append(status);
    for (const value of [run.model_calls, run.tokens.toLocaleString(), (run.model_ms / 1000).toFixed(1) + ' 秒']) row.append(node('td', value));
    const action = node('td'); const button = node('button', '会话', 'text-button'); button.addEventListener('click', () => loadSession(run.session_id).catch(error => toast(error.message))); action.append(button); row.append(action); list.append(row);
  }
  if (!runs.length) { const row = node('tr'); const cell = node('td', '开始一次对话后，这里会保存运行记录。', 'empty-state'); cell.colSpan = 6; row.append(cell); list.append(row); }
}

document.querySelectorAll('[data-view]').forEach(el => el.addEventListener('click', () => showView(el.dataset.view)));
document.querySelectorAll('[data-question]').forEach(el => el.addEventListener('click', () => send(el.dataset.question, el.dataset.mode || 'auto')));
$('new-session').addEventListener('click', async () => { try { const session = await post('/api/sessions', {}); await loadSession(session.id); $('question').focus(); } catch (error) { toast(error.message); } });
$('chat-form').addEventListener('submit', event => { event.preventDefault(); send($('question').value); });
$('question').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); send($('question').value); } });
$('approve').addEventListener('click', () => decide('approve')); $('reject').addEventListener('click', () => decide('reject'));
$('retry').addEventListener('click', async () => { try { state.run = await post(`/api/runs/${state.run.id}/retry`, {}); updateControls(); watchRun(state.run.id); } catch (error) { toast(error.message); } });
$('upload-button').addEventListener('click', () => $('upload-input').click()); $('drop-zone').addEventListener('click', () => $('upload-input').click());
$('drop-zone').addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); $('upload-input').click(); } });
$('upload-input').addEventListener('change', () => upload($('upload-input').files));
for (const name of ['dragenter', 'dragover']) $('drop-zone').addEventListener(name, event => { event.preventDefault(); $('drop-zone').classList.add('dragging'); });
for (const name of ['dragleave', 'drop']) $('drop-zone').addEventListener(name, event => { event.preventDefault(); $('drop-zone').classList.remove('dragging'); });
$('drop-zone').addEventListener('drop', event => upload(event.dataTransfer.files));
$('close-source').addEventListener('click', () => $('source-dialog').close());
$('search-form').addEventListener('submit', event => { event.preventDefault(); previewSearch(); });
$('ask-search').addEventListener('click', () => { showView('chat'); send(previewQuestion, 'answer'); });
(async () => { try { const data = await refresh(); if (data.sessions.length) await loadSession(data.sessions[0].id); else { const session = await post('/api/sessions', {}); await loadSession(session.id); } } catch (error) { toast(error.message); } })();
