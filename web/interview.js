(() => {
  'use strict';
  const questions = window.INTERVIEW_QUESTIONS || [];
  const groupOrder = [...new Set(questions.map(item => item.group))];
  questions.sort((a, b) => groupOrder.indexOf(a.group) - groupOrder.indexOf(b.group));
  const $ = selector => document.querySelector(selector);
  const escape = value => String(value).replace(/[&<>"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
  const groups = [...new Set(questions.map(item => item.group))];
  let group = '全部';
  let mode = 'EASY';
  let completed = {};
  const opened = new Set();
  const formulaCache = new Map();
  function math(tex, display = true) {
    const key = `${display}:${tex}`;
    if (!formulaCache.has(key)) {
      try {
        formulaCache.set(key, window.katex.renderToString(tex, {
          displayMode: display, throwOnError: true, trust: false, strict: 'error', output: 'htmlAndMathml',
        }));
      } catch {
        formulaCache.set(key, `<code class="math-fallback">${escape(tex)}</code>`);
      }
    }
    return `<${display ? 'div' : 'span'} class="${display ? 'math-block' : 'math-inline'}">${formulaCache.get(key)}</${display ? 'div' : 'span'}>`;
  }
  const list = values => values.map(value => `<li>${escape(value)}</li>`).join('');
  try {
    const saved = JSON.parse(localStorage.getItem('llmlab:interview-completed') || '{}');
    if (saved && typeof saved === 'object' && !Array.isArray(saved)) completed = saved;
  } catch { /* Learning content remains available if browser storage is unavailable. */ }
  const saveProgress = () => {
    try { localStorage.setItem('llmlab:interview-completed', JSON.stringify(completed)); }
    catch { /* Progress is a browser convenience, not a prerequisite for learning. */ }
  };
  const workbenchLink = item => item.link ? `/?module=${encodeURIComponent(item.link)}&level=${mode}` : null;
  const sourceLink = item => item.source && /^https:\/\/[^\s]+$/.test(item.source) ? item.source : null;
  const displayCount = () => {
    const n = questions.filter(item => completed[item.id] === true).length;
    $('#completedCount').textContent = `${n} / ${questions.length}`;
    $('#interviewProgress').style.width = `${questions.length ? n / questions.length * 100 : 0}%`;
  };
  const cardHTML = (item, index) => {
    const codeLink = workbenchLink(item);
    const paper = sourceLink(item);
    const expanded = opened.has(item.id);
    const showDerivation = mode === 'EASY';
    const steps = item.derivation.map((text, step) => `<li><span class="derivation-number">${step + 1}</span><div><p>${escape(text)}</p>${item.stepMath?.[step] ? math(item.stepMath[step]) : ''}</div></li>`).join('');
    return `<article class="question-card${expanded ? ' open' : ''}" id="q-${escape(item.id)}" data-id="${escape(item.id)}">
      <button class="question-head" type="button" aria-expanded="${expanded}" aria-controls="body-${escape(item.id)}"><span class="question-number">${String(index+1).padStart(2,'0')}</span><span class="question-main"><span class="question-tags"><span>${escape(item.group)}</span><span>${escape(item.kind)} · ${escape(item.level)}</span></span><h2>${escape(item.title)}</h2><p>${escape(item.question)}</p></span><span class="question-chevron" aria-hidden="true">›</span></button>
      <div class="question-body" id="body-${escape(item.id)}"${expanded ? '' : ' hidden'}>
        <div class="lesson-path" aria-label="学习顺序"><span>01 理解原理</span><span aria-hidden="true">→</span><span>02 数学推导</span><span aria-hidden="true">→</span><span>03 手撕代码</span></div>
        <section class="lesson-stage principle-stage" aria-labelledby="principle-${escape(item.id)}"><h3 id="principle-${escape(item.id)}"><span class="stage-number">01</span>先理解，为什么这样做</h3><p class="principle-copy">${escape(item.principle)}</p>
          <dl class="symbol-grid">${item.symbols.map(s => `<div><dt>${math(s.symbol, false)}</dt><dd>${escape(s.meaning)}</dd></div>`).join('')}</dl>
          <div class="conditions"><h4>适用条件与约定</h4><ul>${list(item.conditions)}</ul></div>
        </section>
        <section class="lesson-stage derivation-stage" aria-labelledby="derivation-${escape(item.id)}"><div class="stage-heading"><h3 id="derivation-${escape(item.id)}"><span class="stage-number">02</span>把数学推导走一遍</h3>${mode === 'HARD' ? `<button class="answer-toggle" type="button" aria-expanded="false" aria-controls="answer-${escape(item.id)}">查看参考推导</button>` : '<span class="stage-note">EASY · 推导已展开</span>'}</div>
          ${mode === 'HARD' ? '<p class="independent-prompt">先根据上面的原理和符号闭卷推导，再展开核对。</p>' : ''}
          <div class="answer-panel" id="answer-${escape(item.id)}"${showDerivation ? '' : ' hidden'}><div class="core-formulas"><h4>核心公式</h4>${item.math.map(tex => math(tex)).join('')}</div><h4>逐步推导</h4><ol class="derivation-steps">${steps}</ol><div class="worked-example"><span>算一个小例子</span><p>${escape(item.example)}</p></div><button class="continue-code" type="button">理解了，开始手撕代码 <span aria-hidden="true">↓</span></button></div>
        </section>
        <details class="lesson-stage coding-stage"><summary><span class="stage-number">03</span><span>把推导写成代码<small>${codeLink ? `${mode} · 在工作台实现` : '独立练习 · 按下方任务实现'}</small></span><span class="coding-chevron" aria-hidden="true">＋</span></summary><div class="coding-content"><h4>实现任务</h4><p>${escape(item.handwrite)}</p><h4>写完后，用这些条件自检</h4><ul class="acceptance-checks">${list(item.checks)}</ul><div class="coding-actions">${codeLink ? `<a class="code-link" href="${escape(codeLink)}">打开 ${mode} 代码练习 ↗</a>` : '<p class="practice-note">此题暂未提供独立代码模板，可在本地 Python 文件中完成。</p>'}</div><h4>面试追问</h4><p>${escape(item.followup)}</p></div></details>
        <div class="question-actions">${paper ? `<a class="source-link" href="${escape(paper)}" target="_blank" rel="noopener noreferrer" title="阅读论文需要联网，题目与公式已在本地">论文与依据 ↗</a>` : ''}<button class="mastered${completed[item.id] ? ' active' : ''}" type="button" aria-pressed="${completed[item.id] ? 'true' : 'false'}">${completed[item.id] ? '✓ 已掌握' : '标记掌握'}</button></div>
      </div></article>`;
  };
  function render() {
    const term = $('#search').value.trim().toLocaleLowerCase();
    const visible = questions.filter(item => (group === '全部' || item.group === group) && (!term || [item.title,item.question,item.group,item.principle,item.handwrite,item.followup,item.formula,...item.derivation,...item.conditions].join(' ').toLocaleLowerCase().includes(term)));
    if (visible.length && !visible.some(item => opened.has(item.id))) opened.add(visible[0].id);
    $('#questionList').innerHTML = visible.map((item,index) => cardHTML(item,index)).join('');
    $('#emptyState').hidden = visible.length > 0;
    $('#resultTitle').textContent = group === '全部' ? '全部题目' : group;
    $('#resultCount').textContent = `共 ${visible.length} 题`;
    document.querySelectorAll('.category').forEach(button => button.classList.toggle('active', button.dataset.group === group));
    displayCount();
  }
  $('#allCount').textContent = questions.length;
  $('#totalBadge').textContent = `${questions.length} 道题`;
  $('#categoryCount').textContent = `${groups.length} 个方向`;
  $('#categories').innerHTML = groups.map(name => `<button type="button" class="category" data-group="${escape(name)}">${escape(name)} <span>${questions.filter(item => item.group === name).length}</span></button>`).join('');
  document.querySelector('.interview-nav').addEventListener('click', event => {
    const button = event.target.closest('.category');
    if (!button) return;
    group = button.dataset.group;
    render();
  });
  $('#search').addEventListener('input', render);
  const setMode = next => {
    mode = next;
    $('#easyMode').classList.toggle('selected', next === 'EASY');
    $('#hardMode').classList.toggle('selected', next === 'HARD');
    $('#easyMode').setAttribute('aria-pressed', String(next === 'EASY'));
    $('#hardMode').setAttribute('aria-pressed', String(next === 'HARD'));
    render();
  };
  $('#easyMode').addEventListener('click', () => setMode('EASY'));
  $('#hardMode').addEventListener('click', () => setMode('HARD'));
  $('#randomQuestion').addEventListener('click', () => {
    group = '全部';
    $('#search').value = '';
    render();
    const item = questions[Math.floor(Math.random() * questions.length)];
    const card = document.getElementById(`q-${item.id}`);
    if (card && !card.classList.contains('open')) card.querySelector('.question-head').click();
    card?.scrollIntoView({behavior:'smooth',block:'center'});
    card?.classList.add('interview-focus');
    setTimeout(() => card?.classList.remove('interview-focus'), 1600);
  });
  $('#questionList').addEventListener('click', event => {
    const card = event.target.closest('.question-card');
    if (!card) return;
    const header = event.target.closest('.question-head');
    if (header) {
      const open = header.getAttribute('aria-expanded') !== 'true';
      header.setAttribute('aria-expanded', String(open));
      card.classList.toggle('open', open);
      card.querySelector('.question-body').hidden = !open;
      if (open) opened.add(card.dataset.id); else opened.delete(card.dataset.id);
      return;
    }
    const answer = event.target.closest('.answer-toggle');
    if (answer) {
      const open = answer.getAttribute('aria-expanded') !== 'true';
      answer.setAttribute('aria-expanded', String(open));
      answer.textContent = open ? '收起参考推导' : '查看参考推导';
      card.querySelector('.answer-panel').hidden = !open;
      card.querySelector('.independent-prompt').hidden = open;
      return;
    }
    if (event.target.closest('.continue-code')) {
      const coding = card.querySelector('.coding-stage');
      coding.open = true;
      coding.scrollIntoView({behavior:'smooth',block:'start'});
      coding.querySelector('summary').focus({preventScroll:true});
      return;
    }
    const mastered = event.target.closest('.mastered');
    if (mastered) {
      completed[card.dataset.id] = !completed[card.dataset.id];
      mastered.classList.toggle('active', completed[card.dataset.id]);
      mastered.setAttribute('aria-pressed', String(completed[card.dataset.id]));
      mastered.textContent = completed[card.dataset.id] ? '✓ 已掌握' : '标记掌握';
      saveProgress();
      displayCount();
    }
  });
  if (location.hash.startsWith('#q-')) opened.add(location.hash.slice(3));
  render();
  if (location.hash.startsWith('#q-')) {
    const card = document.getElementById(location.hash.slice(1));
    card?.scrollIntoView({block:'center'});
  }
})();
