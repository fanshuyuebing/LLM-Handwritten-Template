"""Offline, loopback-only learning workbench. Run trusted local Python only."""
import json
import mimetypes
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
FILES = {str(p.relative_to(ROOT)): p for group in ('Transformer-EASY', 'Transformer-HARD', 'RLHF-EASY', 'RLHF-HARD', 'Interview-EASY', 'Interview-HARD') for p in (ROOT / group).iterdir() if p.suffix in ('.py', '.md')}
FILES['README.md'] = ROOT / 'README.md'
TOKEN = secrets.token_urlsafe(32)
RUN_LOCK = threading.Lock()
# Only explicitly bundled assets are served; arbitrary workspace paths stay private.
VENDOR_ASSETS = {
    '/' + str(path.relative_to(ROOT / 'web')): path
    for path in (ROOT / 'web' / 'vendor').rglob('*')
    if path.is_file() and path.suffix in ('.js', '.css', '.woff', '.woff2', '.ttf')
}

class Handler(BaseHTTPRequestHandler):
    def reply(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.headers.get('Host') not in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'):
            return self.reply({'error': 'Invalid host'}, 403)
        route = urlsplit(self.path).path
        if route == '/api/state':
            return self.reply({'token': TOKEN, 'files': {k: p.read_text() for k, p in FILES.items()}, 'python': sys.version.split()[0]})
        if route in VENDOR_ASSETS:
            path = VENDOR_ASSETS[route]
            data = path.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            return self.wfile.write(data)
        assets = {
            '/': ('index.html', 'text/html'),
            '/index.html': ('index.html', 'text/html'),
            '/style.css': ('style.css', 'text/css'),
            '/app.js': ('app.js', 'text/javascript'),
            '/interview': ('interview.html', 'text/html'),
            '/interview.html': ('interview.html', 'text/html'),
            '/interview.css': ('interview.css', 'text/css'),
            '/interview.js': ('interview.js', 'text/javascript'),
            '/interview-data.js': ('interview-data.js', 'text/javascript'),
        }
        if route not in assets:
            return self.reply({'error': 'Not found'}, 404)
        filename, content_type = assets[route]
        data = (ROOT / 'web' / filename).read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', content_type + '; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.headers.get('X-Workbench-Token') != TOKEN:
            return self.reply({'error': '请刷新页面后重试'}, 403)
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 2_000_000:
                raise ValueError('请求过大或为空')
            body = json.loads(self.rfile.read(size))
            name = body.get('file')
            path = FILES.get(name)
            if path is None or path.suffix != '.py':
                raise ValueError('只允许操作项目练习文件')
            if self.path == '/api/save':
                source = body['source']
                if not isinstance(source, str):
                    raise ValueError('代码必须是文本')
                current = path.read_text()
                if body.get('base') != current:
                    return self.reply({'error': '磁盘文件已在 VS Code 或其他窗口修改。请先点击「从磁盘载入」，避免覆盖新代码。'}, 409)
                backup = ROOT / '.study-backups' / str(time.time_ns()) / name
                backup.parent.mkdir(parents=True, exist_ok=True)
                backup.write_text(current)
                temporary = path.with_suffix('.py.tmp')
                temporary.write_text(source)
                temporary.replace(path)
                return self.reply({'message': '已保存到项目，旧版本已备份'})
            if self.path == '/api/reload':
                return self.reply({'source': path.read_text()})
            if self.path == '/api/open-vscode':
                code = shutil.which('code')
                if not code:
                    return self.reply({'error': '未找到 VS Code 命令行工具 code，请先在 VS Code 中安装 Shell Command。'}, 400)
                subprocess.Popen([code, '--reuse-window', str(path)], cwd=ROOT,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return self.reply({'message': '已在 VS Code 中打开当前练习文件。'})
            if self.path == '/api/run':
                if not RUN_LOCK.acquire(blocking=False):
                    return self.reply({'error': '已有练习正在运行，请稍后再试'}, 409)
                try:
                    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
                    # A file avoids unbounded pipe buffering if a learner writes a noisy loop.
                    import tempfile
                    with tempfile.TemporaryFile() as log:
                        proc = subprocess.Popen([sys.executable, str(path)], cwd=path.parent, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        timed_out = False
                        try:
                            proc.wait(timeout=90)
                        except subprocess.TimeoutExpired:
                            import signal
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                            timed_out = True
                        log.seek(0)
                        output = log.read(200_000).decode(errors='replace')
                    return self.reply({'code': proc.returncode, 'output': output, 'timeout': timed_out})
                finally:
                    RUN_LOCK.release()
            return self.reply({'error': 'Not found'}, 404)
        except (ValueError, KeyError, OSError) as error:
            self.reply({'error': str(error)}, 400)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8765'))
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'离线学习工作台：http://127.0.0.1:{port}  （Ctrl+C 停止）', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
