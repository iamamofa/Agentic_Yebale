#!/usr/bin/env python3
"""Start Streamlit plus a temporary public Cloudflare link. Ctrl+C stops both."""

import argparse
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent
PUBLIC_URL = re.compile(r'https://[a-z0-9-]+\.trycloudflare\.com\b')


def available_port(preferred):
    """Keep existing app processes intact; choose a free port when necessary."""
    with socket.socket() as sock:
        try:
            sock.bind(('127.0.0.1', preferred))
        except OSError:
            sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def healthy(url):
    try:
        with urllib.request.urlopen(url + '/_stcore/health', timeout=4) as response:
            return response.status == 200 and response.read(100).strip() == b'ok'
    except (OSError, urllib.error.URLError):
        return False


def wait_for_server(process, url, timeout=40):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('Streamlit stopped during startup. Check the launcher log in .runtime/.')
        if healthy(url):
            return
        time.sleep(.5)
    raise RuntimeError('Streamlit did not become healthy before the startup timeout.')


def _read_lines(stream, messages, log_path):
    with log_path.open('w') as log:
        for line in stream:
            log.write(line)
            log.flush()
            try:
                messages.put_nowait(line.strip())
            except queue.Full:
                pass


def wait_for_tunnel(process, messages, timeout=75):
    deadline = time.monotonic() + timeout
    url = None
    recent = []
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('The public tunnel stopped. ' + ' '.join(recent[-3:]))
        try:
            line = messages.get(timeout=.5)
        except queue.Empty:
            continue
        recent.append(line)
        recent = recent[-6:]
        match = PUBLIC_URL.search(line)
        if match:
            url = match.group(0)
        if url and 'Registered tunnel connection' in line:
            return url
    raise RuntimeError('Cloudflare could not establish a public connection. Check internet access. '
                       + ' '.join(recent[-3:]))


def stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=6)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8503, help='Preferred local port (default: 8503).')
    parser.add_argument('--no-browser', action='store_true', help='Print the link without opening a browser.')
    parser.add_argument('--local-only', action='store_true', help='Start Streamlit without public sharing.')
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error('Port must be between 0 and 65535.')
    cloudflared = shutil.which('cloudflared')
    if not args.local_only and not cloudflared:
        print('Install Cloudflare Tunnel first: brew install cloudflared', file=sys.stderr)
        return 1
    runtime = ROOT / '.runtime'
    runtime.mkdir(mode=0o700, exist_ok=True)
    state_file = runtime / f'public-{os.getpid()}.json'
    latest_file = runtime / 'public-url.txt'
    port = available_port(args.port)
    local_url = f'http://127.0.0.1:{port}'
    environment = dict(os.environ)
    environment['KNOWING_WHERE_PUBLIC_STATE'] = str(state_file)
    streamlit = tunnel = None
    url = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (runtime / f'streamlit-{os.getpid()}.log').open('w') as log:
            streamlit = subprocess.Popen([
                sys.executable, '-m', 'streamlit', 'run', str(ROOT / 'streamlit_app.py'),
                '--server.address', '127.0.0.1', '--server.port', str(port),
                '--server.headless', 'true', '--server.fileWatcherType', 'none',
            ], cwd=str(ROOT), env=environment, stdout=log, stderr=subprocess.STDOUT)
            print(f'Starting Knowing Where at {local_url} …', flush=True)
            wait_for_server(streamlit, local_url)
            if args.local_only:
                print(f'Local app: {local_url}', flush=True)
                if not args.no_browser:
                    webbrowser.open(local_url)
            else:
                print('Creating your public HTTPS link …', flush=True)
                tunnel = subprocess.Popen([
                    cloudflared, 'tunnel', '--no-autoupdate', '--url', local_url,
                    '--protocol', 'http2',
                ], cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1)
                messages = queue.Queue(maxsize=256)
                threading.Thread(target=_read_lines, args=(tunnel.stdout, messages,
                    runtime / f'cloudflared-{os.getpid()}.log'), daemon=True).start()
                url = wait_for_tunnel(tunnel, messages)
                print(f'Tunnel assigned {url}; verifying public access …', flush=True)
                deadline = time.monotonic() + 90
                while not healthy(url):
                    if tunnel.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError('The tunnel connected, but its public health check did not pass.')
                    time.sleep(1)
                state_file.write_text(json.dumps({'url': url, 'local_port': port,
                                                  'launcher_pid': os.getpid()}))
                latest_file.write_text(url + '\n')
                print(f'\nPUBLIC APP: {url}\nShare this link with anyone.\n'
                      'Keep this terminal and computer running. The link changes after a restart.\n'
                      'Press Ctrl+C to stop the app and its public link.\n', flush=True)
                if not args.no_browser:
                    webbrowser.open(url)
            while streamlit.poll() is None:
                if tunnel is not None and tunnel.poll() is not None:
                    raise RuntimeError('The public tunnel stopped. Restart this launcher for a new link.')
                time.sleep(1)
            raise RuntimeError('The app stopped. Restart this launcher to run it again.')
    except KeyboardInterrupt:
        print('\nStopping the app and public sharing …', flush=True)
        return 0
    except (RuntimeError, OSError) as error:
        print(f'Startup failed: {error}', file=sys.stderr, flush=True)
        return 1
    finally:
        stop(tunnel)
        stop(streamlit)
        state_file.unlink(missing_ok=True)
        if url and latest_file.exists() and latest_file.read_text().strip() == url:
            latest_file.unlink()


if __name__ == '__main__':
    sys.exit(main())
