# 本地公式排版

KaTeX 0.16.22 的构建文件与字体随项目保存，不使用 CDN。页面启动、刷新和公式排版均不需要联网。

- 来源：https://registry.npmjs.org/katex/-/katex-0.16.22.tgz
- 使用方式：https://katex.org/docs/browser.html
- 许可证：MIT，全文在 `katex/LICENSE`

升级时同时替换 JS、CSS 和 fonts，保留许可证，并运行 `node scripts/check_interview.js` 检查题库中全部公式。
