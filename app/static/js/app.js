/* ==========================================================================
 * 教师智能问答系统 — 前端逻辑
 *
 * 对应《教师智能问答系统技术文档》：
 *   - 6.3 节 API 契约（统一错误体 {"error": {code, message, request_id, timestamp}}）
 *   - 6.4 节 SSE 事件协议：start → delta*n → citation → done（异常为 error）
 *
 * ⚠️ 为什么不用浏览器原生 EventSource：
 *   EventSource 只支持 GET，而 /api/chat/stream 是 POST 且需要 JSON 请求体
 *   （技术文档 6.4 已注明）。因此这里用 fetch + ReadableStream 手动解析
 *   `data:` 行，同时获得了「可用 AbortController 中断」的能力。
 *
 * ⚠️ 所有模型输出都必须经过 escapeHtml 再插入 DOM：
 *   大模型的回答内容不可信，直接 innerHTML 会造成 XSS。
 * ========================================================================== */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ */
  /* 常量                                                                */
  /* ------------------------------------------------------------------ */
  const API = {
    session: '/api/session',
    subjects: '/api/subjects',
    chat: '/api/chat',
    chatStream: '/api/chat/stream',
    history: '/api/history',
  };

  const MAX_QUESTION_LENGTH = 2000;   // 与技术文档 6.3.4 的校验规则一致
  const SCROLL_BOTTOM_THRESHOLD = 80; // 距底部多少像素内视为「贴底」

  /* ------------------------------------------------------------------ */
  /* 状态                                                                */
  /* ------------------------------------------------------------------ */
  const state = {
    sessionId: null,
    subjectCode: null,      // null 表示全学科
    subjects: [],
    mode: 'stream',         // 'stream' | 'instant'
    isGenerating: false,
    abortController: null,
    oldestCursor: null,
    hasMore: false,
    isLoadingHistory: false,
  };

  /* ------------------------------------------------------------------ */
  /* DOM 引用                                                            */
  /* ------------------------------------------------------------------ */
  const el = {
    chatArea: document.getElementById('chatArea'),
    messageList: document.getElementById('messageList'),
    welcomeCard: document.getElementById('welcomeCard'),
    exampleList: document.getElementById('exampleList'),
    subjectSelect: document.getElementById('subjectSelect'),
    drawer: document.getElementById('drawer'),
    drawerMask: document.getElementById('drawerMask'),
    drawerSubjects: document.getElementById('drawerSubjects'),
    drawerClose: document.getElementById('drawerClose'),
    menuToggle: document.getElementById('menuToggle'),
    subjectJumpBtn: document.getElementById('subjectJumpBtn'),
    modeButtons: document.querySelectorAll('.mode-btn'),
    input: document.getElementById('questionInput'),
    sendBtn: document.getElementById('sendBtn'),
    stopBtn: document.getElementById('stopBtn'),
    clearBtn: document.getElementById('clearBtn'),
    historyBtn: document.getElementById('historyBtn'),
    loadMoreBtn: document.getElementById('loadMoreBtn'),
    toast: document.getElementById('toast'),
    charCounter: document.getElementById('charCounter'),
  };

  /* ================================================================== */
  /* 工具函数                                                            */
  /* ================================================================== */

  /** HTML 转义：模型输出与用户输入在插入 DOM 前必须转义，防止 XSS。 */
  function escapeHtml(text) {
    return String(text == null ? '' : text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  /**
   * 把纯文本渲染为安全的 HTML：转义后再识别围栏代码块与行内代码。
   * 只支持最必要的两种格式——问答场景里模型最常输出的就是代码。
   */
  function renderText(text) {
    const escaped = escapeHtml(text);
    // 围栏代码块 ```lang ... ```
    let html = escaped.replace(
      /```([a-zA-Z0-9+#-]*)\n?([\s\S]*?)```/g,
      (_, lang, code) =>
        `<pre><code data-lang="${lang || ''}">${code.replace(/\n$/, '')}</code></pre>`
    );
    // 行内代码 `code`（不匹配 <pre> 内的内容）
    html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');
    return html;
  }

  /** 时间戳格式化为 YYYY-MM-DD HH:mm:ss（PRD 5.2.1 的消息展示规范）。 */
  function formatTime(isoString) {
    if (!isoString) return '';
    // 后端返回的是无时区的 ISO 串，直接交给 Date 解析即可（本地时区显示）
    const date = new Date(isoString);
    if (isNaN(date.getTime())) return String(isoString);
    const pad = (n) => String(n).padStart(2, '0');
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ` +
           `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  }

  /** 轻提示（PRD 5.1.3：学科切换提示等）。 */
  let toastTimer = null;
  function showToast(message, isError) {
    el.toast.textContent = message;
    el.toast.classList.toggle('is-error', Boolean(isError));
    el.toast.hidden = false;
    // 强制重排后再加类，保证 transition 生效
    void el.toast.offsetWidth;
    el.toast.classList.add('is-show');

    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      el.toast.classList.remove('is-show');
      setTimeout(() => { el.toast.hidden = true; }, 220);
    }, isError ? 4000 : 2200);
  }

  /** 判断滚动容器是否已经贴在底部（决定要不要自动滚动）。 */
  function isNearBottom() {
    const area = el.chatArea;
    return area.scrollHeight - area.scrollTop - area.clientHeight < SCROLL_BOTTOM_THRESHOLD;
  }

  function scrollToBottom(smooth) {
    el.chatArea.scrollTo({
      top: el.chatArea.scrollHeight,
      behavior: smooth ? 'smooth' : 'auto',
    });
  }

  /**
   * 统一 API 请求封装：自动解析后端的技术文档 6.1 统一错误体。
   * @throws {Error} 带 code 属性的错误对象
   */
  async function api(path, options) {
    const response = await fetch(path, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });

    if (!response.ok) {
      let code = 'HTTP_' + response.status;
      let message = `请求失败（HTTP ${response.status}）`;
      try {
        const body = await response.json();
        if (body && body.error) {
          code = body.error.code || code;
          message = body.error.message || message;
        }
      } catch (_) {
        // 响应体不是 JSON（如网关错误页），沿用默认提示
      }
      const error = new Error(message);
      error.code = code;
      throw error;
    }

    if (response.status === 204) return null;
    return response.json();
  }

  /* ================================================================== */
  /* 渲染：消息气泡与溯源                                                */
  /* ================================================================== */

  /**
   * 创建一条消息行。
   * @param {'user'|'bot'} role
   * @param {Object} options { text, references, time, isThinking }
   * @returns {{row: HTMLElement, bubble: HTMLElement, meta: HTMLElement}}
   */
  function createMessageRow(role, options) {
    const opts = options || {};
    const isUser = role === 'user';

    const row = document.createElement('div');
    row.className = 'message-row ' + (isUser ? 'is-user' : 'is-bot');

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = isUser ? '我' : '马';
    avatar.setAttribute('aria-hidden', 'true');

    const wrap = document.createElement('div');
    wrap.className = 'bubble-wrap';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';

    if (opts.isThinking) {
      // PRD 4.2.2：提交问题后显示「正在思考...」
      bubble.innerHTML =
        '<span class="thinking">正在思考' +
        '<i class="thinking-dot"></i><i class="thinking-dot"></i><i class="thinking-dot"></i>' +
        '</span>';
    } else if (isUser) {
      // 用户输入用 textContent，天然安全
      bubble.textContent = opts.text || '';
    } else {
      bubble.innerHTML = renderText(opts.text || '');
    }

    const meta = document.createElement('div');
    meta.className = 'bubble-meta';
    if (opts.time) meta.textContent = formatTime(opts.time);

    wrap.appendChild(bubble);

    if (opts.references && opts.references.length) {
      wrap.appendChild(createCitations(opts.references));
    }
    wrap.appendChild(meta);

    row.appendChild(avatar);
    row.appendChild(wrap);
    el.messageList.appendChild(row);

    return { row, bubble, meta };
  }

  /**
   * 渲染溯源信息卡片（PRD 5.2.2：
   * 默认显示学科标签 +「参考资料N」，点击展开摘要）。
   */
  function createCitations(references) {
    const box = document.createElement('div');
    box.className = 'citations';

    references.forEach((ref) => {
      const item = document.createElement('div');
      item.className = 'citation-item';

      const head = document.createElement('button');
      head.type = 'button';
      head.className = 'citation-head';

      const tag = document.createElement('span');
      tag.className = 'citation-tag';
      tag.textContent = ref.subject_name || '参考资料';

      const title = document.createElement('span');
      title.className = 'citation-title';
      title.textContent = `参考资料${ref.index || ''}`;

      const score = document.createElement('span');
      score.className = 'citation-score';
      score.textContent = typeof ref.score === 'number' ? ref.score.toFixed(3) : '';

      const arrow = document.createElement('span');
      arrow.className = 'citation-arrow';
      arrow.textContent = '▾';

      head.append(tag, title, score, arrow);

      const body = document.createElement('div');
      body.className = 'citation-body';

      const source = document.createElement('div');
      source.className = 'citation-source';
      // 页码为 -1 / null 时不显示（如 DOCX 无页码）
      const pageText = ref.page_no ? ` 第${ref.page_no}页` : '';
      source.textContent = `来源：${ref.source || '未知'}${pageText}`;

      const summary = document.createElement('div');
      summary.textContent = ref.summary || '（无摘要）';

      body.append(source, summary);
      item.append(head, body);

      head.addEventListener('click', () => item.classList.toggle('is-open'));
      box.appendChild(item);
    });

    return box;
  }

  /** 隐藏欢迎卡片（开始对话后不再需要）。 */
  function hideWelcomeCard() {
    if (el.welcomeCard && !el.welcomeCard.hidden) {
      el.welcomeCard.hidden = true;
    }
  }

  /** 切换生成中的 UI 状态：禁用输入、显示停止按钮。 */
  function setGenerating(generating) {
    state.isGenerating = generating;
    el.input.disabled = generating;
    el.stopBtn.hidden = !generating;
    el.sendBtn.hidden = generating;
    updateSendButtonState();
  }

  /* ================================================================== */
  /* 会话与学科                                                          */
  /* ================================================================== */

  /** 初始化：创建会话并加载学科列表。 */
  async function bootstrap() {
    bindEvents();

    try {
      const session = await api(API.session, {
        method: 'POST',
        body: JSON.stringify({}),
      });
      state.sessionId = session.session_id;

      // 用会话接口带回的学科列表直接渲染，省一次请求
      if (Array.isArray(session.subjects) && session.subjects.length) {
        state.subjects = session.subjects.map((item) => ({
          code: item.code,
          name: item.name,
        }));
        renderSubjects();
      } else {
        await loadSubjects();
      }
    } catch (error) {
      showToast('会话初始化失败：' + error.message, true);
      el.input.disabled = true;
      el.sendBtn.disabled = true;
    }
  }

  /** 单独获取学科列表（含 doc_count，用于展示知识覆盖度）。 */
  async function loadSubjects() {
    try {
      const data = await api(API.subjects, { method: 'GET' });
      state.subjects = (data.subjects || []).map((item) => ({
        code: item.code,
        name: item.name,
        docCount: item.doc_count,
      }));
      renderSubjects();
    } catch (error) {
      showToast('学科列表加载失败：' + error.message, true);
    }
  }

  /** 渲染学科下拉与抽屉。 */
  function renderSubjects() {
    // ---- 顶栏下拉 ----
    el.subjectSelect.innerHTML = '';
    state.subjects.forEach((item) => {
      const option = document.createElement('option');
      option.value = item.code == null ? '' : item.code;
      option.textContent = item.name;
      el.subjectSelect.appendChild(option);
    });
    el.subjectSelect.value = state.subjectCode == null ? '' : state.subjectCode;

    // ---- 移动端抽屉 ----
    el.drawerSubjects.innerHTML = '';
    state.subjects.forEach((item) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'drawer-subject';
      const code = item.code == null ? null : item.code;
      if (code === state.subjectCode) button.classList.add('is-active');
      button.textContent = item.name +
        (item.docCount ? `（${item.docCount} 条知识）` : '');
      button.addEventListener('click', () => {
        applySubject(code, item.name);
        closeDrawer();
      });
      el.drawerSubjects.appendChild(button);
    });
  }

  /** 应用学科切换并提示（PRD 5.3.2：Toast 提示「已切换到XX学科」）。 */
  function applySubject(code, name) {
    state.subjectCode = code;
    el.subjectSelect.value = code == null ? '' : code;
    renderSubjects();
    showToast(`已切换到${name}`);
  }

  function openDrawer() {
    el.drawer.classList.add('is-open');
    el.drawer.setAttribute('aria-hidden', 'false');
    el.drawerMask.hidden = false;
    el.menuToggle.setAttribute('aria-expanded', 'true');
  }

  function closeDrawer() {
    el.drawer.classList.remove('is-open');
    el.drawer.setAttribute('aria-hidden', 'true');
    el.drawerMask.hidden = true;
    el.menuToggle.setAttribute('aria-expanded', 'false');
  }

  /* ================================================================== */
  /* 问答                                                                */
  /* ================================================================== */

  /** 发送问题（按当前模式分发到流式 / 即时）。 */
  async function sendQuestion() {
    const question = el.input.value.trim();

    // PRD 4.2.3：空输入提示
    if (!question) {
      showToast('请输入你的问题');
      el.input.focus();
      return;
    }
    if (question.length > MAX_QUESTION_LENGTH) {
      showToast(`问题不能超过 ${MAX_QUESTION_LENGTH} 字符`);
      return;
    }
    if (!state.sessionId) {
      showToast('会话尚未就绪，请刷新页面重试', true);
      return;
    }
    if (state.isGenerating) return;

    hideWelcomeCard();

    // 清空输入并复位高度
    el.input.value = '';
    autoResizeInput();
    updateSendButtonState();
    updateCharCounter();

    // 渲染用户气泡（PRD 5.2.3 步骤 2）
    createMessageRow('user', { text: question, time: new Date().toISOString() });
    scrollToBottom(true);

    if (state.mode === 'stream') {
      await sendByStream(question);
    } else {
      await sendByInstant(question);
    }
  }

  /**
   * 流式问答：POST /api/chat/stream + 手动解析 SSE。
   * 事件协议见技术文档 6.4。
   */
  async function sendByStream(question) {
    setGenerating(true);

    // 助手气泡先显示「正在思考...」（PRD 5.2.3 步骤 3）
    const node = createMessageRow('bot', { isThinking: true });
    let answerText = '';
    let hasContent = false;
    scrollToBottom(true);

    state.abortController = new AbortController();

    try {
      const response = await fetch(API.chatStream, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: state.sessionId,
          question: question,
          subject_code: state.subjectCode,
          use_history: true,
        }),
        signal: state.abortController.signal,
      });

      if (!response.ok) {
        // 准备阶段的错误（参数非法、会话不存在）会走这里
        let message = `请求失败（HTTP ${response.status}）`;
        try {
          const body = await response.json();
          if (body && body.error) message = body.error.message;
        } catch (_) { /* 非 JSON 响应，沿用默认提示 */ }
        throw new Error(message);
      }
      if (!response.body) {
        throw new Error('当前浏览器不支持流式响应');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      // SSE 以空行分隔事件；跨 chunk 的事件需要靠 buffer 拼接
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        let boundary;
        while ((boundary = buffer.indexOf('\n\n')) >= 0) {
          const rawEvent = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          const event = parseSseEvent(rawEvent);
          if (!event) continue;

          switch (event.type) {
            case 'start':
              // 后端已建立记录，此处无需处理（前端不依赖 record_id 渲染）
              break;

            case 'delta': {
              if (!hasContent) {
                // 首个增量到达：清掉「正在思考」，开始打字机效果
                node.bubble.innerHTML = '';
                hasContent = true;
              }
              answerText += event.content || '';
              node.bubble.innerHTML = renderText(answerText) + '<i class="cursor-blink"></i>';
              if (isNearBottom()) scrollToBottom(false);
              break;
            }

            case 'citation':
              if (Array.isArray(event.references) && event.references.length) {
                // 插到 meta 之前，保持「正文 → 溯源 → 时间戳」的顺序
                node.row.querySelector('.bubble-wrap')
                  .insertBefore(createCitations(event.references), node.meta);
              }
              break;

            case 'done':
              node.meta.textContent = formatTime(new Date().toISOString());
              break;

            case 'error':
              // 后端已推送部分内容后失败：保留内容并提示错误
              if (!hasContent) node.bubble.innerHTML = '';
              node.bubble.innerHTML +=
                `<div class="citation-source" style="color:var(--danger)">` +
                `${escapeHtml(event.message || '生成失败')}</div>`;
              break;

            default:
              break;
          }
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        // 用户主动中断（PRD 2.2.2.2：支持中断和重新提问）
        showToast('已停止生成');
        if (!hasContent) node.bubble.textContent = '（已停止）';
      } else {
        if (!hasContent) node.bubble.innerHTML = '';
        node.bubble.innerHTML +=
          `<div class="citation-source" style="color:var(--danger)">` +
          `${escapeHtml(error.message)}</div>`;
        showToast(error.message, true);
      }
    } finally {
      // 收尾：去掉闪烁光标，补上时间戳
      const cursor = node.bubble.querySelector('.cursor-blink');
      if (cursor) cursor.remove();
      if (!node.meta.textContent) node.meta.textContent = formatTime(new Date().toISOString());
      state.abortController = null;
      setGenerating(false);
      scrollToBottom(true);
      el.input.focus();
    }
  }

  /** 即时问答：POST /api/chat，一次性返回完整答案（PRD 2.2.2.1）。 */
  async function sendByInstant(question) {
    setGenerating(true);
    const node = createMessageRow('bot', { isThinking: true });
    scrollToBottom(true);

    try {
      const data = await api(API.chat, {
        method: 'POST',
        body: JSON.stringify({
          session_id: state.sessionId,
          question: question,
          subject_code: state.subjectCode,
          use_history: true,
        }),
      });

      const wrap = node.row.querySelector('.bubble-wrap');
      if (data.references && data.references.length) {
        wrap.insertBefore(createCitations(data.references), node.meta);
      }
      node.bubble.innerHTML = renderText(data.answer || '（无回答）');
      node.meta.textContent =
        `${formatTime(new Date().toISOString())}` +
        (data.is_from_cache ? ' · 缓存命中' : '') +
        (data.response_time_ms ? ` · ${data.response_time_ms}ms` : '');
    } catch (error) {
      node.bubble.innerHTML = '';
      node.bubble.innerHTML =
        `<div class="citation-source" style="color:var(--danger)">` +
        `${escapeHtml(error.message)}</div>`;
      node.meta.textContent = formatTime(new Date().toISOString());
      showToast(error.message, true);
    } finally {
      setGenerating(false);
      scrollToBottom(true);
      el.input.focus();
    }
  }

  /** 解析单条 SSE 事件文本（可能包含多行，取 data: 之后的内容）。 */
  function parseSseEvent(rawEvent) {
    const dataLines = rawEvent
      .split('\n')
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trim());

    if (!dataLines.length) return null;
    try {
      return JSON.parse(dataLines.join(''));
    } catch (_) {
      return null;
    }
  }

  /** 停止生成（PRD 2.2.2.2 支持中断）。 */
  function stopGenerating() {
    if (state.abortController) {
      state.abortController.abort();
    }
  }

  /* ================================================================== */
  /* 历史记录                                                            */
  /* ================================================================== */

  /**
   * 加载历史（PRD 5.4：按时间倒序、支持滚动加载更多）。
   * @param {boolean} reset true=重新加载最新一页并替换视图；false=加载更早的一页
   */
  async function loadHistory(reset) {
    if (state.isLoadingHistory || !state.sessionId) return;
    if (!reset && !state.hasMore) return;

    state.isLoadingHistory = true;
    el.loadMoreBtn.disabled = true;
    el.loadMoreBtn.textContent = '加载中...';

    const keepScrollHeight = el.chatArea.scrollHeight;
    const keepScrollTop = el.chatArea.scrollTop;

    try {
      const params = new URLSearchParams({ session_id: state.sessionId, limit: '20' });
      if (!reset && state.oldestCursor != null) {
        params.set('cursor', String(state.oldestCursor));
      }

      const data = await api(`${API.history}?${params.toString()}`, { method: 'GET' });
      const records = data.records || [];

      state.hasMore = Boolean(data.has_more);
      state.oldestCursor = records.length
        ? records[records.length - 1].record_id
        : state.oldestCursor;

      if (reset) {
        // 重新查看历史：清空当前视图，按时间正序渲染持久化记录
        el.messageList.innerHTML = '';
        hideWelcomeCard();
      }

      if (!records.length && reset) {
        showToast('当前会话还没有历史记录');
      }

      // 接口返回的是倒序（最新在前），插入时反转成正序才符合阅读顺序
      const fragment = document.createDocumentFragment();
      records.slice().reverse().forEach((record) => {
        fragment.appendChild(buildHistoryUserRow(record));
        fragment.appendChild(buildHistoryBotRow(record));
      });

      if (reset) {
        el.messageList.appendChild(fragment);
        scrollToBottom(false);
      } else {
        // 加载更早的记录：插到最前面，并保持视觉位置不跳动
        el.messageList.insertBefore(fragment, el.messageList.firstChild);
        el.chatArea.scrollTop = keepScrollTop + (el.chatArea.scrollHeight - keepScrollHeight);
      }

      el.loadMoreBtn.hidden = !state.hasMore;
      el.loadMoreBtn.textContent = state.hasMore ? '加载更早的对话' : '已加载全部历史';
    } catch (error) {
      showToast('历史加载失败：' + error.message, true);
    } finally {
      state.isLoadingHistory = false;
      el.loadMoreBtn.disabled = false;
      if (el.loadMoreBtn.textContent === '加载中...') {
        el.loadMoreBtn.textContent = '加载更早的对话';
      }
    }
  }

  /** 由历史记录构造用户消息行。 */
  function buildHistoryUserRow(record) {
    const node = createMessageRow('user', {
      text: record.question,
      time: record.created_at,
    });
    return node.row;
  }

  /** 由历史记录构造助手消息行（含溯源）。 */
  function buildHistoryBotRow(record) {
    const node = createMessageRow('bot', {
      text: record.answer || '',
      references: record.references || [],
      time: record.created_at,
    });
    return node.row;
  }

  /** 清空历史（PRD 2.1.2.3：清空操作需有确认提示）。 */
  async function clearHistory() {
    if (!state.sessionId) return;

    // 浏览器原生 confirm 足以满足「确认提示」的要求，且无需额外组件
    if (!window.confirm('确定要清空对话历史吗？此操作不可撤销。')) return;

    try {
      await api(`${API.history}?session_id=${encodeURIComponent(state.sessionId)}`, {
        method: 'DELETE',
      });
      el.messageList.innerHTML = '';
      el.welcomeCard.hidden = false;
      state.hasMore = false;
      state.oldestCursor = null;
      el.loadMoreBtn.hidden = true;
      showToast('对话历史已清空');
    } catch (error) {
      showToast('清空失败：' + error.message, true);
    }
  }

  /* ================================================================== */
  /* 输入框行为                                                          */
  /* ================================================================== */

  /** 自适应高度：内容变多时长高，到上限后内部滚动（PRD 5.5 移动端友好）。 */
  function autoResizeInput() {
    el.input.style.height = 'auto';
    el.input.style.height = Math.min(el.input.scrollHeight, 160) + 'px';
  }

  /** 未输入时禁用发送按钮（PRD 5.1.3）。 */
  function updateSendButtonState() {
    const hasText = el.input.value.trim().length > 0;
    el.sendBtn.disabled = !hasText || state.isGenerating || !state.sessionId;
  }

  /** 字符计数（上限 2000，与技术文档 6.3.4 的校验规则一致）。 */
  function updateCharCounter() {
    const length = el.input.value.length;
    el.charCounter.textContent = `${length} / ${MAX_QUESTION_LENGTH}`;
    el.charCounter.style.color =
      length > MAX_QUESTION_LENGTH ? 'var(--danger)' : '';
  }

  /* ================================================================== */
  /* 事件绑定                                                            */
  /* ================================================================== */
  function bindEvents() {
    // ---- 发送 ----
    el.sendBtn.addEventListener('click', sendQuestion);
    el.stopBtn.addEventListener('click', stopGenerating);

    // 回车发送、Shift+回车换行（PRD 5.1.3）
    el.input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        if (!el.sendBtn.disabled) sendQuestion();
      }
    });

    el.input.addEventListener('input', () => {
      autoResizeInput();
      updateSendButtonState();
      updateCharCounter();
    });

    // ---- 学科选择 ----
    el.subjectSelect.addEventListener('change', (event) => {
      const code = event.target.value || null;
      const subject = state.subjects.find((item) =>
        (item.code == null ? null : item.code) === code);
      applySubject(code, subject ? subject.name : '全学科');
    });

    // 移动端抽屉
    el.menuToggle.addEventListener('click', openDrawer);
    el.drawerClose.addEventListener('click', closeDrawer);
    el.drawerMask.addEventListener('click', closeDrawer);
    el.subjectJumpBtn.addEventListener('click', openDrawer);
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') closeDrawer();
    });

    // ---- 回答模式 ----
    el.modeButtons.forEach((button) => {
      button.addEventListener('click', () => {
        state.mode = button.dataset.mode;
        el.modeButtons.forEach((item) =>
          item.classList.toggle('is-active', item === button));
        showToast(state.mode === 'stream' ? '已切换到流式回答' : '已切换到即时回答');
      });
    });

    // ---- 示例问题 ----
    el.exampleList.addEventListener('click', (event) => {
      const chip = event.target.closest('.example-chip');
      if (!chip) return;
      el.input.value = chip.textContent.trim();
      autoResizeInput();
      updateSendButtonState();
      updateCharCounter();
      sendQuestion();
    });

    // ---- 历史与清空 ----
    el.historyBtn.addEventListener('click', () => loadHistory(true));
    el.loadMoreBtn.addEventListener('click', () => loadHistory(false));
    el.clearBtn.addEventListener('click', clearHistory);

    // 滚动到顶部附近时自动加载更早的历史（PRD 5.4.2）
    el.chatArea.addEventListener('scroll', () => {
      if (el.chatArea.scrollTop < 60 && state.hasMore && !state.isLoadingHistory) {
        loadHistory(false);
      }
    });

    // 页面隐藏（切到后台）时中断生成，避免无谓消耗模型配额
    document.addEventListener('visibilitychange', () => {
      if (document.hidden && state.isGenerating) {
        stopGenerating();
      }
    });
  }

  /* ================================================================== */
  /* 启动                                                                */
  /* ================================================================== */
  document.addEventListener('DOMContentLoaded', () => {
    updateSendButtonState();
    updateCharCounter();
    bootstrap();
  });
})();
