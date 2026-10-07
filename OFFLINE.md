# 火车上的离线学习

工作台无需 GPU、模型权重、API Key 或联网服务。首次在新电脑安装 Python 和依赖需要联网；出发前完成安装即可离线学习。

## 首次安装（新电脑）

先安装 Python 3.12，在克隆后的项目根目录执行：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python scripts/check_env.py
```

上面是 macOS / Linux 命令。Windows 创建环境后，使用 `.venv\Scripts\python.exe` 代替 `.venv/bin/python`，通过 `.venv\Scripts\python.exe web/server.py` 启动页面。前端资源已随仓库保存，无需安装 Node.js 或 npm。

本机已有 Python 3.12.12、PyTorch 2.14.0 和 NumPy，可直接使用下面的启动命令。

## 启动页面

```bash
cd LLM-Handwritten-Template  # 替换为你的项目路径
./start.sh
```

用 Safari / Chrome 打开 http://127.0.0.1:8765 。保持终端窗口运行，停止时按 Ctrl+C。端口被占用可用 `PORT=8766 ./start.sh` 并访问对应端口。

页面提供六个模块、EASY/HARD 切换、34 道面试推导题、排版后的 Markdown 学习说明、带行号和语法高亮的代码编辑、保存并运行、笔记和掌握标记。无需 Node.js 或 npm。

长文件可以点击编辑器右上角的「专注编辑」占满窗口，用 Esc 退出；「定位 TODO」会跳到下一个待实现位置。代码字号可用 A− / A+ 调节。编辑区默认高度已加大。

安装 VS Code 且 `code` 命令可用时，页面下方的「在 VS Code 打开」会直接打开当前练习文件。修改后先在 VS Code 保存，再回到页面点击「从磁盘载入」，页面会读取最新内容，无需复制粘贴。如果两个地方同时修改，页面保存时会阻止覆盖磁盘上的新版本。

- 保存直接更新对应练习文件，旧内容保存在 `.study-backups/时间戳/原路径`，可手动复制恢复。
- 笔记自动保存在当前浏览器的 localStorage；更换浏览器、端口或清除浏览器数据后不会保留，请把重要笔记另存文件。
- 运行的是本机 Python，90 秒后自动停止；较长训练请在终端运行。页面运行仅展示脚本结果，不会自动验证每个 TODO 是否正确。
- 模板保留了待实现的 TODO。出现 `NotImplementedError` 是正常的练习提示，不代表环境安装失败。
- 请避免同时在多个页面或编辑器里修改同一个文件。

## 环境自检

```bash
.venv/bin/python scripts/check_env.py
```

自检会禁止 Python socket 联网，验证 CPU 运算、反向传播和全部十二个练习文件的导入。它不等同于算法正确性测试。

直接运行练习：

```bash
.venv/bin/python RLHF-EASY/rlhf_env.py
.venv/bin/python Transformer-EASY/transformer.py
.venv/bin/python RLHF-EASY/ppo.py
```

建议从 Transformer-EASY 开始；RLHF 先做环境里的 TODO-A/B/C，再做算法。

## 离线重装（本机）

保留 `.python/`、`.venv/` 和 `wheelhouse/`。这些目录不会提交到 Git，但已经下载到这台电脑。单独 git clone 不会包含它们。

依赖缺失时可完全离线安装：

```bash
.venv/bin/python -m pip install --no-index --find-links=wheelhouse -r requirements.lock.txt
```

如果 `.venv` 被删除，在当前项目根目录执行：

```bash
.python/cpython-3.12.12-macos-aarch64-none/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install --no-index --find-links=wheelhouse -r requirements.lock.txt
```

离线安装包适用于当前 Apple Silicon Mac / Python 3.12，不能直接用于 Windows 或 Intel Mac。保持当前项目路径最省事，移动项目后需重建虚拟环境。
