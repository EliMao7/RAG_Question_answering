/* ==========================================================================
 * 前端集成自检
 *
 * 为什么需要它：前端是纯静态 JS，浏览器里点一遍才能发现问题；而装 Playwright
 * 要额外下载几百 MB 浏览器。这里用一个**最小 DOM 桩**在 Node 中真实执行
 * app/static/js/app.js，并连到运行中的后端跑完整链路：
 *
 *   DOM 元素清单从 index.html 解析得到（保证 id 与真实页面一致）
 *   fetch / AbortController / TextDecoder 用 Node 原生实现
 *   事件监听器被记录下来，测试里手动触发（等价于用户点击）
 *
 * 这样能真实覆盖：会话创建 → 发送问题 → SSE 解析 → 渲染更新 → 历史加载 → 清空。
 *
 * 前置条件：后端已启动
 *     uvicorn app.main:app --port 8000
 *
 * 用法::
 *     node scripts/frontend_check.js [baseUrl]
 * ========================================================================== */
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const BASE_URL = process.argv[2] || 'http://127.0.0.1:8000';
const ROOT = path.resolve(__dirname, '..');
const APP_JS = path.join(ROOT, 'app', 'static', 'js', 'app.js');
const INDEX_HTML = path.join(ROOT, 'app', 'static', 'index.html');

const results = [];
function check(name, ok, detail) {
  results.push({ name, ok });
  console.log(`  ${ok ? '[OK]  ' : '[FAIL]'} ${name}${detail ? ' — ' + detail : ''}`);
}

/* ------------------------------------------------------------------ */
/* 最小 DOM 实现                                                       */
/* ------------------------------------------------------------------ */

class ClassList {
  constructor(el) { this.el = el; this.set = new Set(); }
  add(...names) { names.forEach((n) => n && this.set.add(n)); this._sync(); }
  remove(...names) { names.forEach((n) => this.set.delete(n)); this._sync(); }
  contains(name) { return this.set.has(name); }
  toggle(name, force) {
    const on = force === undefined ? !this.set.has(name) : Boolean(force);
    if (on) this.set.add(name); else this.set.delete(name);
    this._sync();
    return on;
  }
  _sync() {
    // 把 class 列表同步回 className，便于 querySelector 查找
    this.el._className = [...this.set].join(' ');
  }
}

class StubElement {
  constructor(tagName, id) {
    this.tagName = (tagName || 'div').toUpperCase();
    this.id = id || '';
    this.childNodes = [];
    this.parentNode = null;
    this.listeners = {};
    this.attributes = {};
    this.style = {};
    this.dataset = {};
    this.hidden = false;
    this.disabled = false;
    this.value = '';
    this.type = '';
    this._className = '';
    this._innerHTML = '';
    this._textContent = '';

    // 布局相关属性（供滚动逻辑读取）
    this.scrollTop = 0;
    this.scrollHeight = 1000;
    this.clientHeight = 500;

    this.classList = new ClassList(this);
  }

  get className() { return this._className; }
  set className(v) {
    this._className = v || '';
    this.classList.set = new Set(this._className.split(/\s+/).filter(Boolean));
  }

  /* innerHTML 作为不透明字符串保存：app.js 只对由它自己 createElement
     出来的节点做 querySelector，因此不需要解析 innerHTML。 */
  get innerHTML() { return this._innerHTML; }
  set innerHTML(v) { this._innerHTML = String(v); }

  get textContent() { return this._textContent || this._innerHTML; }
  set textContent(v) { this._textContent = String(v); }

  appendChild(node) {
    if (node && node.__isFragment) {
      node.childNodes.forEach((child) => this.appendChild(child));
      node.childNodes = [];
      return node;
    }
    node.parentNode = this;
    this.childNodes.push(node);
    return node;
  }

  insertBefore(node, ref) {
    const index = ref ? this.childNodes.indexOf(ref) : -1;
    if (index >= 0) {
      node.parentNode = this;
      this.childNodes.splice(index, 0, node);
    } else {
      this.appendChild(node);
    }
    return node;
  }

  removeChild(node) {
    const index = this.childNodes.indexOf(node);
    if (index >= 0) this.childNodes.splice(index, 1);
    return node;
  }

  remove() { if (this.parentNode) this.parentNode.removeChild(this); }

  querySelector(selector) { return this._find(selector, false)[0] || null; }
  querySelectorAll(selector) { return this._find(selector, true); }

  _find(selector, all) {
    const found = [];
    const match = (node) => {
      if (selector.startsWith('.')) return node.classList.contains(selector.slice(1));
      if (selector.startsWith('#')) return node.id === selector.slice(1);
      return node.tagName === selector.toUpperCase();
    };
    const walk = (node) => {
      for (const child of node.childNodes) {
        if (match(child)) {
          found.push(child);
          if (!all) return true;
        }
        if (walk(child) && !all) return true;
      }
      return false;
    };
    walk(this);
    return found;
  }

  get firstChild() { return this.childNodes[0] || null; }

  // 事件
  addEventListener(type, handler) {
    (this.listeners[type] = this.listeners[type] || []).push(handler);
  }
  dispatch(type, event) {
    (this.listeners[type] || []).forEach((handler) => handler(event || {}));
  }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  getAttribute(key) { return this.attributes[key]; }
  focus() { this.focused = true; }
  closest(selector) {
    let node = this;
    while (node) {
      if (selector.startsWith('.') && node.classList &&
          node.classList.contains(selector.slice(1))) return node;
      node = node.parentNode;
    }
    return null;
  }
  scrollTo() { this.scrolled = true; }
}

class StubFragment extends StubElement {
  constructor() { super('#fragment'); this.__isFragment = true; }
}

/* 从 index.html 解析出真实存在的 id 与示例按钮数量，保证桩与页面一致 */
function buildElements() {
  const html = fs.readFileSync(INDEX_HTML, 'utf8');
  const ids = [...html.matchAll(/id="([^"]+)"/g)].map((m) => m[1]);
  const elements = new Map();
  ids.forEach((id) => elements.set(id, new StubElement('div', id)));

  // 页面里带 id 的元素类型修正（select/textarea 需要正确属性）
  if (elements.has('subjectSelect')) elements.get('subjectSelect').tagName = 'SELECT';
  if (elements.has('questionInput')) elements.get('questionInput').tagName = 'TEXTAREA';

  // chatArea 是滚动容器
  const chatArea = elements.get('chatArea');
  chatArea.scrollHeight = 1200;
  chatArea.clientHeight = 500;
  chatArea.scrollTop = 0;

  return { elements, ids, html };
}

const { elements, ids, html } = buildElements();

const document = {
  getElementById: (id) => elements.get(id) || null,
  querySelectorAll: (selector) => {
    if (selector === '.mode-btn') {
      const stream = new StubElement('button');
      stream.dataset.mode = 'stream';
      stream.classList.add('mode-btn', 'is-active');
      const instant = new StubElement('button');
      instant.dataset.mode = 'instant';
      instant.classList.add('mode-btn');
      return [stream, instant];
    }
    return [];
  },
  createElement: (tag) => new StubElement(tag),
  createDocumentFragment: () => new StubFragment(),
  addEventListener: (type, handler) => {
    (document._listeners = document._listeners || {});
    (document._listeners[type] = document._listeners[type] || []).push(handler);
  },
  dispatch: (type, event) => {
    ((document._listeners || {})[type] || []).forEach((h) => h(event || {}));
  },
  hidden: false,
};

const window = {
  confirm: () => true,        // 清空历史时自动确认
  setTimeout,
  clearTimeout,
  fetch,
  URLSearchParams,
  Date,
  console,
};

/* ------------------------------------------------------------------ */
/* 载入并执行被测代码                                                  */
/* ------------------------------------------------------------------ */
const sandbox = {
  document,
  window,
  console,
  fetch,
  AbortController,
  TextDecoder,
  URLSearchParams,
  setTimeout,
  clearTimeout,
  Date,
  JSON,
  Math,
  Promise,
  Error,
  isNaN,
  String,
  Number,
  Boolean,
  Array,
  Object,
  RegExp,
  Set,
  Map,
};
sandbox.globalThis = sandbox;
// Node 原生 fetch 是相对路径无法解析的，这里包一层拼上 baseUrl
sandbox.fetch = (input, init) => {
  const url = typeof input === 'string' && input.startsWith('/')
    ? BASE_URL + input
    : input;
  return fetch(url, init);
};
sandbox.window.fetch = sandbox.fetch;

vm.createContext(sandbox);
const source = fs.readFileSync(APP_JS, 'utf8');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* ------------------------------------------------------------------ */
/* 测试主流程                                                          */
/* ------------------------------------------------------------------ */
(async function main() {
  console.log('='.repeat(70));
  console.log(`前端集成自检 → ${BASE_URL}`);
  console.log('='.repeat(70));

  // ---- 0. 静态检查：HTML 元素与 JS 的 id 引用是否对得上 ----
  console.log('\n[0] DOM 契约检查（HTML id ↔ JS 引用）');
  const jsSource = source;
  const referencedIds = [...new Set([...jsSource.matchAll(/getElementById\('([^']+)'\)/g)]
    .map((m) => m[1]))];
  const missing = referencedIds.filter((id) => !ids.includes(id));
  check(`JS 引用的 ${referencedIds.length} 个 id 均存在于 HTML`, missing.length === 0,
        missing.length ? '缺失：' + missing.join(', ') : '');
  check('示例问题按钮已渲染', (html.match(/example-chip/g) || []).length >= 4,
        `${(html.match(/example-chip/g) || []).length} 个`);

  // ---- 1. 执行 app.js ----
  console.log('\n[1] 执行 app.js');
  try {
    vm.runInContext(source, sandbox, { filename: 'app.js' });
    check('脚本加载无异常', true);
  } catch (error) {
    check('脚本加载无异常', false, `${error.name}: ${error.message}`);
    return finish();
  }

  // 触发 DOMContentLoaded（等价于页面加载完成）
  try {
    document.dispatch('DOMContentLoaded');
    check('DOMContentLoaded 处理无异常', true);
  } catch (error) {
    check('DOMContentLoaded 处理无异常', false, `${error.name}: ${error.message}`);
    return finish();
  }

  // 等待 bootstrap 创建会话
  await sleep(2500);
  const sendBtn = elements.get('sendBtn');
  const input = elements.get('questionInput');
  const subjectSelect = elements.get('subjectSelect');
  const messageList = elements.get('messageList');
  const toast = elements.get('toast');

  check('会话初始化后发送按钮可用',
        sendBtn.disabled === true, '未输入时保持禁用（符合 PRD 5.1.3）');
  check('学科下拉已填充选项',
        subjectSelect.childNodes.length >= 6,
        `${subjectSelect.childNodes.length} 个选项`);

  // ---- 2. 空输入不应发送 ----
  console.log('\n[2] 空输入校验（PRD 4.2.3）');
  input.value = '   ';
  input.dispatch('input');
  check('空白输入时发送按钮禁用', sendBtn.disabled === true);

  // ---- 3. 问候语（走轻量路径）----
  console.log('\n[3] 发送「你好」→ 应走问候语分流');
  input.value = '你好';
  input.dispatch('input');
  check('有输入后发送按钮启用', sendBtn.disabled === false);

  const before = messageList.childNodes.length;
  sendBtn.dispatch('click');
  await sleep(3000);

  const afterGreeting = messageList.childNodes.length;
  check('新增了用户与助手两条消息', afterGreeting - before === 2,
        `从 ${before} 增至 ${afterGreeting}`);
  const greetingBot = messageList.childNodes[afterGreeting - 1];
  const greetingText = greetingBot.childNodes[1].childNodes[0].innerHTML || '';
  check('助手回复了问候语', greetingText.includes('你好'),
        greetingText.slice(0, 40).replace(/<[^>]*>/g, ''));

  // ---- 4. 流式问答（核心链路）----
  console.log('\n[4] 流式问答（SSE 打字机）');
  input.value = '什么是Transformer？';
  input.dispatch('input');
  const beforeStream = messageList.childNodes.length;
  sendBtn.dispatch('click');

  await sleep(2000);
  const midCount = messageList.childNodes.length;
  check('已插入助手气泡（思考中）', midCount - beforeStream === 2);
  check('生成中显示停止按钮', elements.get('stopBtn').hidden === false);
  check('生成中输入框被禁用', input.disabled === true);

  await sleep(25000);
  const afterStream = messageList.childNodes.length;
  const streamBot = messageList.childNodes[afterStream - 1];
  const bubbleWrap = streamBot.childNodes[1];
  const streamText = (bubbleWrap.childNodes[0].innerHTML || '').replace(/<[^>]*>/g, '');

  check('流式回答有内容', streamText.length > 50, `${streamText.length} 字符`);
  check('生成结束恢复发送按钮', elements.get('stopBtn').hidden === true);
  check('生成结束恢复输入框', input.disabled === false);
  check('助手气泡出现溯源卡片',
        bubbleWrap.childNodes.some((n) => n.classList.contains('citations')));
  check('消息带时间戳', Boolean(bubbleWrap.childNodes[bubbleWrap.childNodes.length - 1].textContent));
  console.log(`       → ${streamText.slice(0, 60)}…`);

  // ---- 5. 学科切换 ----
  console.log('\n[5] 学科切换（PRD 5.3.2）');
  subjectSelect.value = 'ai';
  subjectSelect.dispatch('change', { target: { value: 'ai' } });
  check('切换学科后弹出提示', toast.textContent.includes('已切换到'),
        toast.textContent);

  // ---- 6. 即时问答模式 ----
  console.log('\n[6] 即时问答模式（PRD 2.2.2.1）');
  const modeBtns = document.querySelectorAll('.mode-btn');
  modeBtns[1].dispatch('click');           // 切到 instant
  await sleep(200);
  input.value = '课程大纲包含哪些阶段？';
  input.dispatch('input');
  const beforeInstant = messageList.childNodes.length;
  sendBtn.dispatch('click');
  await sleep(30000);

  const afterInstant = messageList.childNodes.length;
  const instantBot = messageList.childNodes[afterInstant - 1];
  const instantText = (instantBot.childNodes[1].childNodes[0].innerHTML || '')
    .replace(/<[^>]*>/g, '');
  check('即时模式返回完整答案', instantText.length > 50, `${instantText.length} 字符`);
  check('新增两条消息', afterInstant - beforeInstant === 2);

  // ---- 7. 历史加载 ----
  console.log('\n[7] 历史加载（游标分页）');
  elements.get('historyBtn').dispatch('click');
  await sleep(2500);
  check('历史记录已渲染',
        messageList.childNodes.length > 0,
        `${messageList.childNodes.length} 条消息行`);

  // ---- 8. 清空历史 ----
  console.log('\n[8] 清空历史（PRD 2.1.2.3）');
  elements.get('clearBtn').dispatch('click');
  await sleep(2000);
  check('消息列表已清空', messageList.childNodes.length === 0,
        `${messageList.childNodes.length} 条`);
  check('欢迎卡片重新显示', elements.get('welcomeCard').hidden === false);
  check('清空后弹出提示', toast.textContent.includes('清空'), toast.textContent);

  // ---- 9. 移动端抽屉 ----
  console.log('\n[9] 移动端抽屉交互');
  elements.get('menuToggle').dispatch('click');
  check('抽屉打开', elements.get('drawer').classList.contains('is-open'));
  check('遮罩显示', elements.get('drawerMask').hidden === false);
  elements.get('drawerClose').dispatch('click');
  check('抽屉关闭', !elements.get('drawer').classList.contains('is-open'));

  finish();
})().catch((error) => {
  console.error('\n测试执行异常：', error);
  process.exit(1);
});

function finish() {
  const failed = results.filter((r) => !r.ok);
  console.log('\n' + '='.repeat(70));
  console.log(`通过 ${results.length - failed.length} / 失败 ${failed.length}`);
  if (failed.length) {
    console.log('\n失败项：');
    failed.forEach((f) => console.log(`  - ${f.name}`));
  }
  console.log('='.repeat(70));
  process.exit(failed.length ? 1 : 0);
}
