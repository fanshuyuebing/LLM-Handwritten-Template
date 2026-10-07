/* Validate every locally stored equation using the exact browser renderer. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');
const katex = require(path.join(root, 'web/vendor/katex/katex.min.js'));
const sandbox = {window: {}};
vm.runInNewContext(fs.readFileSync(path.join(root, 'web/interview-data.js'), 'utf8'), sandbox);
const questions = sandbox.window.INTERVIEW_QUESTIONS;
const links = new Set(['transformer','env','ppo','dpo','grpo','opd']);
assert.equal(questions.length, 34);
assert.equal(new Set(questions.map(q => q.id)).size, questions.length);
let formulaCount = 0;
for (const q of questions) {
  for (const field of ['principle','example','handwrite','question']) assert.ok(q[field]?.length, `${q.id}: ${field}`);
  for (const field of ['math','symbols','conditions','checks','derivation']) assert.ok(q[field]?.length, `${q.id}: ${field}`);
  if (q.link) assert.ok(links.has(q.link), `${q.id}: invalid workbench link`);
  if (q.stepMath) assert.equal(q.stepMath.length, q.derivation.length, `${q.id}: step equations must align`);
  // Catch accidental Python/JS string escapes such as \b, \t and \r in TeX.
  const tex = [...q.math, ...q.symbols.map(s => s.symbol), ...(q.stepMath || []).filter(Boolean)];
  for (const expression of tex) {
    assert.ok(!/[\x00-\x1f]/.test(expression), `${q.id}: control character in ${JSON.stringify(expression)}`);
    const rendered = katex.renderToString(expression, {throwOnError:true, strict:'error', trust:false});
    assert.ok(rendered.includes('katex-mathml'), `${q.id}: missing accessible math`);
    formulaCount++;
  }
}
console.log(`✓ ${questions.length} 道题内容完整，${formulaCount} 个公式/符号均可由本地 KaTeX 严格排版。`);
console.log('✓ 题目 ID 唯一，原理/推导/代码任务完整，练习跳转模块有效。');
