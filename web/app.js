(() => {
  'use strict';

  const $ = selector => document.querySelector(selector);
  const modules = [
    { title: 'Transformer', file: 'transformer.py', short: '注意力与自回归生成', description: 'Decoder-Only Transformer · 从 RMSNorm 到 KV Cache' },
    { title: 'RLHF 环境', file: 'rlhf_env.py', short: '搭建强化学习基础', description: 'Tokenizer、数据收集器与 log probabilities' },
    { title: 'PPO', file: 'ppo.py', short: '在线采样与策略优化', description: '在线采样、GAE 与裁剪目标' },
    { title: 'DPO', file: 'dpo.py', short: '从偏好数据中学习', description: '偏好数据与 Bradley–Terry 损失' },
    { title: 'GRPO', file: 'grpo.py', short: '组内相对优势估计', description: '组内相对优势与 KL 正则' },
    { title: 'OPD', file: 'opd.py', short: '学生轨迹上的教师监督', description: 'On-Policy Distillation · 学生自采样与逐 token KL' },
  ];
  let state = null;
  let moduleIndex = 0;
  const deepLink = new URLSearchParams(location.search);
  const linkedModule = ['transformer', 'env', 'ppo', 'dpo', 'grpo', 'opd'].indexOf(deepLink.get('module'));
  if (linkedModule >= 0) moduleIndex = linkedModule;
  if (['EASY', 'HARD'].includes(deepLink.get('level'))) $('#level').value = deepLink.get('level');
  let activeTab = 'code';
  let dirty = false;
  let busy = false;
  let ready = false;
  let previousLevel = $('#level').value;
  let editorSize = 14;

  const fileKey = (index = moduleIndex, level = $('#level').value) =>
    `${index === 0 ? 'Transformer' : index === 5 ? 'Interview' : 'RLHF'}-${level}/${modules[index].file}`;
  const allKeys = modules.flatMap((_, index) => ['EASY', 'HARD'].map(level => fileKey(index, level)));
  const status = (message, kind = 'info') => {
    $('#status').textContent = message;
    $('#status').dataset.kind = kind;
  };
  const store = {
    get(key, fallback = '') {
      try { return localStorage.getItem(`llmlab:${key}`) ?? fallback; }
      catch { return fallback; }
    },
    set(key, value) {
      try {
        localStorage.setItem(`llmlab:${key}`, value);
        return true;
      } catch {
        status('浏览器存储不可用，请复制笔记到本地文件保存。', 'error');
        return false;
      }
    },
  };
  const savedEditorSize = Number(store.get('editor-size', '14'));
  if (Number.isFinite(savedEditorSize)) editorSize = Math.min(20, Math.max(11, savedEditorSize));
  function setEditorSize(size) {
    editorSize = Math.min(20, Math.max(11, size));
    $('#codePane').style.setProperty('--editor-size', `${editorSize}px`);
    $('#codePane').style.setProperty('--editor-leading', `${Math.round(editorSize * 1.9)}px`);
    store.set('editor-size', String(editorSize));
    syncScroll();
  }
  $('#fontDown').addEventListener('click', () => setEditorSize(editorSize - 1));
  $('#fontUp').addEventListener('click', () => setEditorSize(editorSize + 1));
  setEditorSize(editorSize);

  function toggleFocus(force) {
    const focused = typeof force === 'boolean' ? force : !document.body.classList.contains('editor-focus');
    document.body.classList.toggle('editor-focus', focused);
    $('#focusEditor').textContent = focused ? '退出专注' : '专注编辑';
    $('#focusEditor').setAttribute('aria-pressed', String(focused));
    if (focused) $('#editor').focus();
    requestAnimationFrame(syncScroll);
  }
  $('#focusEditor').addEventListener('click', () => toggleFocus());
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && document.body.classList.contains('editor-focus')) toggleFocus(false);
  });
  $('#nextTodo').addEventListener('click', () => {
    const editor = $('#editor');
    const locations = [...editor.value.matchAll(/^\s*raise NotImplementedError/gm)].map(match => match.index);
    if (!locations.length) { status('当前文件没有待实现的 TODO。'); return; }
    const next = locations.find(index => index > editor.selectionStart) ?? locations[0];
    const line = editor.value.slice(0, next).split('\n').length;
    editor.focus();
    editor.setSelectionRange(next, next);
    editor.scrollTop = Math.max(0, (line - 6) * Math.round(editorSize * 1.9));
    cursorPosition();
    syncScroll();
  });
  let completed = {};
  try {
    const saved = JSON.parse(store.get('completed', '{}'));
    if (saved && typeof saved === 'object' && !Array.isArray(saved)) {
      completed = Object.fromEntries(allKeys.map(key => [key, saved[key] === true]));
    }
  } catch { /* Invalid browser storage must not prevent the workbench from opening. */ }

  const escapeHTML = value => String(value).replace(/[&<>"']/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[character]);

  // Render only a small, local Markdown vocabulary. Source HTML is always text.
  function inlineMarkdown(source, depth = 0) {
    if (depth > 8) return escapeHTML(source);
    const pattern = /(`+)([^`]*?)\1|\[([^\]\n]+)\]\(([^\s)]+)\)|\*\*([^\n]+?)\*\*|__([^\n]+?)__|\*([^*\n]+)\*|_([^_\n]+)_|~~([^\n]+?)~~/g;
    let result = '';
    let offset = 0;
    for (const match of source.matchAll(pattern)) {
      result += escapeHTML(source.slice(offset, match.index));
      if (match[1]) result += `<code>${escapeHTML(match[2])}</code>`;
      else if (match[3]) {
        let href = null;
        try {
          const url = new URL(match[4]);
          if (url.protocol === 'http:' || url.protocol === 'https:') href = url.href;
        } catch { /* Unsupported or relative links are displayed as plain text. */ }
        result += href
          ? `<a href="${escapeHTML(href)}" target="_blank" rel="noopener noreferrer">${inlineMarkdown(match[3], depth + 1)}</a>`
          : inlineMarkdown(match[3], depth + 1);
      } else if (match[5] || match[6]) result += `<strong>${inlineMarkdown(match[5] || match[6], depth + 1)}</strong>`;
      else if (match[7] || match[8]) result += `<em>${inlineMarkdown(match[7] || match[8], depth + 1)}</em>`;
      else result += `<del>${inlineMarkdown(match[9], depth + 1)}</del>`;
      offset = match.index + match[0].length;
    }
    return result + escapeHTML(source.slice(offset));
  }

  function markdown(source) {
    const lines = String(source || '').replace(/\r\n?/g, '\n').split('\n');
    const listPattern = /^(\s*)([-+*]|\d+[.)])\s+(.*)$/;
    const tableCells = line => line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(cell => cell.trim());
    const isTableDivider = line => /\|/.test(line || '') && tableCells(line).every(cell => /^:?-{3,}:?$/.test(cell));
    const isBlock = index => /^(?:\s*$|#{1,6}\s|\s*```|\s*~~~|\s*>|\s*(?:[-*_]\s*){3,}$)/.test(lines[index])
      || listPattern.test(lines[index]) || isTableDivider(lines[index + 1]);

    function parseList(start, indent) {
      const first = lines[start].match(listPattern);
      const ordered = /^\d/.test(first[2]);
      const tag = ordered ? 'ol' : 'ul';
      const startNumber = ordered ? Number.parseInt(first[2], 10) : 1;
      let html = `<${tag}${ordered && startNumber !== 1 ? ` start="${startNumber}"` : ''}>`;
      let index = start;
      while (index < lines.length) {
        const current = lines[index].match(listPattern);
        if (!current || current[1].length !== indent || /^\d/.test(current[2]) !== ordered) break;
        html += `<li>${inlineMarkdown(current[3])}`;
        index += 1;
        while (index < lines.length) {
          const nested = lines[index].match(listPattern);
          if (nested && nested[1].length > indent) {
            const child = parseList(index, nested[1].length);
            html += child.html;
            index = child.index;
          } else if (nested || !lines[index].trim() || /^\s*/.exec(lines[index])[0].length <= indent) break;
          else {
            html += `<br>${inlineMarkdown(lines[index].trim())}`;
            index += 1;
          }
        }
        html += '</li>';
        if (!lines[index]?.trim() && lines[index + 1]?.match(listPattern)?.[1].length === indent) index += 1;
      }
      return { html: `${html}</${tag}>`, index };
    }

    let html = '';
    let index = 0;
    while (index < lines.length) {
      const line = lines[index];
      if (!line.trim()) { index += 1; continue; }
      const fence = line.match(/^\s*(`{3,}|~{3,})(.*)$/);
      if (fence) {
        const code = [];
        index += 1;
        const closing = new RegExp(`^\\s*${fence[1][0]}{${fence[1].length},}\\s*$`);
        while (index < lines.length && !closing.test(lines[index])) code.push(lines[index++]);
        if (index < lines.length) index += 1;
        const language = fence[2].trim().replace(/[^\w-]/g, '');
        html += `<pre><code${language ? ` class="language-${language}"` : ''}>${escapeHTML(code.join('\n'))}</code></pre>`;
        continue;
      }
      const heading = line.match(/^(#{1,6})\s+(.+?)\s*#*$/);
      if (heading) {
        html += `<h${heading[1].length}>${inlineMarkdown(heading[2])}</h${heading[1].length}>`;
        index += 1;
        continue;
      }
      if (/^\s*(?:[-*_]\s*){3,}$/.test(line)) { html += '<hr>'; index += 1; continue; }
      if (/^\s*>/.test(line)) {
        const quote = [];
        while (index < lines.length && /^\s*>/.test(lines[index])) quote.push(lines[index++].replace(/^\s*>\s?/, ''));
        html += `<blockquote>${markdown(quote.join('\n'))}</blockquote>`;
        continue;
      }
      if (line.includes('|') && isTableDivider(lines[index + 1])) {
        const headers = tableCells(line);
        const alignments = tableCells(lines[index + 1]).map(cell => cell.startsWith(':') && cell.endsWith(':') ? 'center' : cell.endsWith(':') ? 'right' : 'left');
        const cells = (row, tag) => headers.map((_, cell) => `<${tag} class="align-${alignments[cell] || 'left'}">${inlineMarkdown(row[cell] || '')}</${tag}>`).join('');
        html += `<div class="table-wrap"><table><thead><tr>${cells(headers, 'th')}</tr></thead><tbody>`;
        index += 2;
        while (index < lines.length && lines[index].trim() && lines[index].includes('|')) html += `<tr>${cells(tableCells(lines[index++]), 'td')}</tr>`;
        html += '</tbody></table></div>';
        continue;
      }
      const list = line.match(listPattern);
      if (list) {
        const parsed = parseList(index, list[1].length);
        html += parsed.html;
        index = parsed.index;
        continue;
      }
      const paragraph = [line.trim()];
      index += 1;
      while (index < lines.length && !isBlock(index)) paragraph.push(lines[index++].trim());
      html += `<p>${inlineMarkdown(paragraph.join('\n')).replace(/\n/g, '<br>')}</p>`;
    }
    return html;
  }

  function highlightPython(source) {
    const tokens = /"""[\s\S]*?(?:"""|$)|'''[\s\S]*?(?:'''|$)|"(?:\\[\s\S]|[^"\\\n])*"?|'(?:\\[\s\S]|[^'\\\n])*'?|#[^\n]*|\b(?:False|None|True|and|as|assert|async|await|break|class|continue|def|del|elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|not|or|pass|raise|return|try|while|with|yield)\b|\b(?:0[xX][\da-fA-F]+|0[bB][01]+|0[oO][0-7]+|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)(?:j)?\b/g;
    let result = '';
    let offset = 0;
    for (const match of source.matchAll(tokens)) {
      const token = match[0];
      const kind = token[0] === '#' ? 'comment' : /["']/.test(token[0]) ? 'string' : /^\d/.test(token) ? 'number' : 'keyword';
      result += escapeHTML(source.slice(offset, match.index));
      result += `<span class="tok-${kind}">${escapeHTML(token)}</span>`;
      offset = match.index + token.length;
    }
    return result + escapeHTML(source.slice(offset)) + '\n';
  }

  function syncScroll() {
    const editor = $('#editor');
    const lineNumbers = $('#editor-lines');
    const highlight = $('#codeHighlight');
    if (lineNumbers) lineNumbers.scrollTop = editor.scrollTop;
    if (highlight) {
      highlight.scrollTop = editor.scrollTop;
      highlight.scrollLeft = editor.scrollLeft;
    }
  }

  function cursorPosition() {
    const display = $('#cursorPosition');
    if (!display) return;
    const prefix = $('#editor').value.slice(0, $('#editor').selectionStart);
    const line = prefix.split('\n').length;
    const column = prefix.length - prefix.lastIndexOf('\n');
    display.textContent = `Ln ${line}, Col ${column}`;
  }

  function updateEditor() {
    const editor = $('#editor');
    const lineNumbers = $('#editor-lines');
    if (lineNumbers) lineNumbers.textContent = Array.from({ length: editor.value.split('\n').length }, (_, index) => index + 1).join('\n') + '\n';
    if ($('#codeHighlight')) {
      $('#codeHighlight').innerHTML = highlightPython(editor.value);
      if ($('#codePane')) $('#codePane').classList.add('highlight-enabled');
    }
    cursorPosition();
    syncScroll();
  }

  function updateCounts() {
    $('#todos').textContent = ( $('#editor').value.match(/raise NotImplementedError/g) || [] ).length;
    const mastered = allKeys.filter(key => completed[key]).length;
    $('#progress').textContent = `${mastered} / ${allKeys.length}`;
    $('#complete').textContent = completed[fileKey()] ? '✓ 已掌握' : '标记掌握';
    $('#complete').classList.toggle('is-complete', Boolean(completed[fileKey()]));
    $('#complete').setAttribute('aria-pressed', String(Boolean(completed[fileKey()])));
    if ($('#progressFill')) $('#progressFill').style.width = `${mastered / allKeys.length * 100}%`;
    if ($('#sidebarProgress')) $('#sidebarProgress').textContent = `${mastered} / ${allKeys.length}`;
    document.querySelectorAll('.nav').forEach((button, index) => {
      const count = ['EASY', 'HARD'].filter(level => completed[fileKey(index, level)]).length;
      const marker = button.querySelector('.nav-check');
      if (marker) marker.textContent = count === 2 ? '✓' : `${count}/2`;
      button.classList.toggle('is-complete', count === 2);
    });
  }

  function setRunState(label, kind = 'idle') {
    const runState = $('#runState');
    if (runState) { runState.textContent = label; runState.dataset.state = kind; }
  }

  function selectTab(name) {
    if (name !== 'code' && document.body.classList.contains('editor-focus')) toggleFocus(false);
    activeTab = name;
    document.querySelectorAll('[data-tab]').forEach(button => {
      const selected = button.dataset.tab === name;
      button.classList.toggle('active', selected);
      button.setAttribute('aria-selected', String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
    const panes = { code: $('#codePane') || $('#editor'), docs: $('#docs'), notes: $('#notesPane') || $('#notes') };
    Object.entries(panes).forEach(([key, pane]) => {
      if (pane) pane.classList.toggle('hidden', key !== name);
    });
    // Keep these children visible when the surrounding pane owns tab visibility.
    if ($('#codePane')) $('#editor').classList.remove('hidden');
    if ($('#notesPane')) $('#notes').classList.remove('hidden');
    if (ready) $('#saveLabel').textContent = name === 'notes' ? '笔记自动保存在此浏览器' : name === 'docs' ? '本地学习文档' : dirty ? '尚未保存' : '本地文件';
    $('#save').disabled = !ready || busy || name !== 'code';
    $('#run').disabled = !ready || busy || name !== 'code';
    $('#openVscode').disabled = !ready || busy || name !== 'code';
    $('#reloadCode').disabled = !ready || busy || name !== 'code';
    $('#focusEditor').disabled = name !== 'code';
    $('#nextTodo').disabled = name !== 'code';
    if (name === 'code') requestAnimationFrame(syncScroll);
  }

  function render() {
    const key = fileKey();
    const current = modules[moduleIndex];
    $('#heading').textContent = current.title;
    $('#description').textContent = current.description;
    if ($('#moduleSubtitle')) $('#moduleSubtitle').textContent = current.title;
    if ($('#moduleNumber')) $('#moduleNumber').textContent = String(moduleIndex + 1).padStart(2, '0');
    if ($('#difficultyHint')) $('#difficultyHint').textContent = $('#level').value === 'EASY' ? '带实现提示，逐步理解核心原理' : '独立推导，把理解写成代码';
    $('#filename').textContent = key;
    $('#editor').value = state.files[key];
    $('#editor').scrollTop = 0;
    $('#editor').scrollLeft = 0;
    $('#editor').setSelectionRange(0, 0);
    $('#docs').innerHTML = markdown(state.files[`${key.split('/')[0]}/README.md`] || '暂无学习说明。');
    $('#docs').scrollTop = 0;
    $('#notes').value = store.get(`notes:${key}`);
    $('#notes').scrollTop = 0;
    dirty = false;
    $('#output').textContent = '运行结果将在这里显示。写下第一行代码，开始今天的练习。';
    setRunState('等待运行');
    status('');
    document.querySelectorAll('.nav').forEach((button, index) => {
      button.classList.toggle('active', index === moduleIndex);
      if (index === moduleIndex) button.setAttribute('aria-current', 'page');
      else button.removeAttribute('aria-current');
    });
    updateEditor();
    updateCounts();
    selectTab(activeTab);
  }

  async function api(route, body) {
    const response = await fetch(`/api/${route}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Workbench-Token': state.token },
      body: JSON.stringify(body),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `请求失败（${response.status}）`);
    return data;
  }

  async function saveCode() {
    const key = fileKey();
    const source = $('#editor').value;
    const result = await api('save', { file: key, source, base: state.files[key] });
    state.files[key] = source;
    dirty = false;
    $('#saveLabel').textContent = '已保存到本地';
    status(result.message || '代码已保存。', 'success');
  }

  function controls(isBusy) {
    busy = isBusy;
    const disabled = !ready || busy;
    document.querySelectorAll('#save, #run, #openVscode, #reloadCode, #level, #complete, .nav').forEach(element => { element.disabled = disabled; });
    $('#editor').readOnly = disabled;
    $('#notes').readOnly = !ready;
    $('#save').disabled = disabled || activeTab !== 'code';
    $('#run').disabled = disabled || activeTab !== 'code';
    $('#openVscode').disabled = disabled || activeTab !== 'code';
    $('#reloadCode').disabled = disabled || activeTab !== 'code';
    $('#run').setAttribute('aria-busy', String(busy));
  }

  $('#save').addEventListener('click', async () => {
    if (!ready || busy || activeTab !== 'code') return;
    controls(true);
    try { await saveCode(); }
    catch (error) { status(error.message, 'error'); }
    finally { controls(false); }
  });

  $('#openVscode').addEventListener('click', async () => {
    if (!ready || busy || activeTab !== 'code') return;
    controls(true);
    try {
      if (dirty) await saveCode();
      const result = await api('open-vscode', { file: fileKey() });
      status(result.message + ' 编辑完成后，回到页面点击「从磁盘载入」。', 'success');
    } catch (error) { status(error.message, 'error'); }
    finally { controls(false); }
  });

  $('#reloadCode').addEventListener('click', async () => {
    if (!ready || busy || activeTab !== 'code' || !canSwitch()) return;
    controls(true);
    try {
      const result = await api('reload', { file: fileKey() });
      state.files[fileKey()] = result.source;
      const position = $('#editor').selectionStart;
      $('#editor').value = result.source;
      $('#editor').setSelectionRange(Math.min(position, result.source.length), Math.min(position, result.source.length));
      dirty = false;
      $('#saveLabel').textContent = '已从磁盘载入';
      updateEditor();
      updateCounts();
      status('已载入 VS Code 中保存的最新代码。', 'success');
    } catch (error) { status(error.message, 'error'); }
    finally { controls(false); }
  });

  $('#run').addEventListener('click', async () => {
    if (!ready || busy || activeTab !== 'code') return;
    controls(true);
    setRunState('正在运行', 'running');
    try {
      await saveCode();
      $('#output').textContent = '正在使用本地 Python 运行，请稍候……';
      status('运行中，可先查看学习说明。');
      const result = await api('run', { file: fileKey() });
      const output = String(result.output || '');
      $('#output').textContent = output || '（无输出）';
      if (result.timeout) {
        status('运行超时，进程已停止。', 'error');
        setRunState('运行超时', 'error');
      } else if (result.code === 0) {
        status('脚本正常结束。是否掌握请结合输出自行判断。', 'success');
        setRunState('运行完成', 'success');
      } else if (output.includes('NotImplementedError')) {
        status('已运行到待实现的 TODO，补全后继续练习。', 'info');
        setRunState('待实现 TODO', 'pending');
      } else {
        status(`运行失败，退出码 ${result.code}，请查看输出。`, 'error');
        setRunState('运行失败', 'error');
      }
    } catch (error) {
      status(error.message, 'error');
      $('#output').textContent = `无法完成运行：${error.message}`;
      setRunState('运行失败', 'error');
    } finally { controls(false); }
  });

  $('#editor').addEventListener('input', () => {
    dirty = $('#editor').value !== state?.files[fileKey()];
    $('#saveLabel').textContent = dirty ? '尚未保存' : '本地文件';
    updateEditor();
    updateCounts();
  });
  $('#editor').addEventListener('scroll', syncScroll);
  ['click', 'keyup', 'select'].forEach(event => $('#editor').addEventListener(event, cursorPosition));
  $('#editor').addEventListener('keydown', event => {
    if (event.key !== 'Tab' || $('#editor').readOnly) return;
    event.preventDefault();
    const editor = event.target;
    editor.setRangeText('    ', editor.selectionStart, editor.selectionEnd, 'end');
    editor.dispatchEvent(new Event('input'));
  });
  $('#notes').addEventListener('input', () => {
    if (ready && store.set(`notes:${fileKey()}`, $('#notes').value)) $('#saveLabel').textContent = '笔记已自动保存';
  });
  $('#complete').addEventListener('click', () => {
    if (!ready || busy) return;
    completed[fileKey()] = !completed[fileKey()];
    if (store.set('completed', JSON.stringify(completed))) status(completed[fileKey()] ? '已记录掌握，继续下一步吧。' : '已取消掌握标记。', 'success');
    updateCounts();
  });
  const canSwitch = () => !dirty || window.confirm('当前代码尚未保存。切换后将丢弃这些修改，确定继续？');
  $('#level').addEventListener('change', () => {
    if (!ready || busy || !canSwitch()) { $('#level').value = previousLevel; return; }
    previousLevel = $('#level').value;
    render();
  });
  const tabButtons = [...document.querySelectorAll('[data-tab]')];
  tabButtons.forEach((button, index) => {
    button.addEventListener('click', () => selectTab(button.dataset.tab));
    button.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabButtons.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabButtons.length) % tabButtons.length;
      tabButtons[next].focus();
      selectTab(tabButtons[next].dataset.tab);
    });
  });
  window.addEventListener('beforeunload', event => {
    if (!dirty) return;
    event.preventDefault();
    event.returnValue = '';
  });
  document.addEventListener('keydown', event => {
    if (!(event.metaKey || event.ctrlKey) || event.key.toLowerCase() !== 's') return;
    event.preventDefault();
    if (!ready || busy) return;
    if (activeTab === 'notes') {
      if (store.set(`notes:${fileKey()}`, $('#notes').value)) status('学习笔记已保存在此浏览器。', 'success');
    } else if (activeTab === 'code') $('#save').click();
  });

  controls(false);
  selectTab('code');
  setRunState('等待运行');
  status('正在连接本地学习环境……');
  fetch('/api/state')
    .then(response => {
      if (!response.ok) throw new Error('本地服务不可用');
      return response.json();
    })
    .then(data => {
      if (!data.files || !allKeys.every(key => typeof data.files[key] === 'string')) throw new Error('练习文件不完整，请检查项目目录');
      state = data;
      $('#python').textContent = data.python;
      modules.forEach((module, index) => {
        const button = document.createElement('button');
        button.className = 'nav';
        button.type = 'button';
        button.innerHTML = `<span class="nav-index">${String(index + 1).padStart(2, '0')}</span><span class="nav-copy"><span class="nav-title">${escapeHTML(module.title)}</span><span class="nav-description">${escapeHTML(module.short)}</span></span><span class="nav-check" aria-hidden="true">0/2</span>`;
        button.addEventListener('click', () => {
          if (busy || index === moduleIndex || !canSwitch()) return;
          moduleIndex = index;
          render();
        });
        $('#navigation').append(button);
      });
      ready = true;
      render();
      controls(false);
    })
    .catch(error => {
      ready = false;
      controls(false);
      status(`连接失败：${error.message}。请确认启动终端仍在运行，然后刷新页面。`, 'error');
      setRunState('环境未连接', 'error');
    });
})();
