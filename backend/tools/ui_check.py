"""Headless UI smoke test driven over the Chrome DevTools protocol.

Renders the dashboard in headless Chrome, waits for the live WebSocket
snapshot, collects any console errors / uncaught exceptions, prints a few
DOM facts and writes a PNG screenshot.

Usage:  python tools/ui_check.py http://127.0.0.1:51917/ out.png
"""
import asyncio
import base64
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request

import websockets

CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]
CHROME = next((p for p in CANDIDATES if os.path.exists(p)), CANDIDATES[-1])

# Keep every temp artefact (browser profile, screenshots) inside the project
# folder so the system drive is never used.
TMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       ".tmp")
os.makedirs(TMP_DIR, exist_ok=True)
tempfile.tempdir = TMP_DIR


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


PORT = free_port()


async def drive(ws_url, screenshot_path, pre_shot_js=None, url=None):
    events = []
    async with websockets.connect(ws_url, max_size=40_000_000) as ws:
        seq = 0

        async def call(method, params=None, timeout=25):
            nonlocal seq
            seq += 1
            my_id = seq
            await ws.send(json.dumps({"id": my_id, "method": method,
                                      "params": params or {}}))
            deadline = time.time() + timeout
            while time.time() < deadline:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                msg = json.loads(raw)
                if msg.get("id") == my_id:
                    return msg
                events.append(msg)
            raise TimeoutError(method)

        await call("Runtime.enable")
        await call("Log.enable")
        await call("Page.enable")
        if url:
            try:
                await call("Runtime.evaluate",
                           {"expression": f"location.replace({json.dumps(url)})"},
                           timeout=10)
            except Exception as exc:  # noqa: BLE001 - page may already be loading
                print("navigate warning:", repr(exc))
        await asyncio.sleep(9)

        expr = """(() => ({
            ready: document.readyState,
            camItems: document.querySelectorAll('.cam-item').length,
            kpis: [...document.querySelectorAll('.kpi .k-value')].map(e => e.textContent.trim()),
            zones: [...document.querySelectorAll('.zone-row')].map(r => r.querySelector('.zone-name').textContent),
            alerts: document.querySelectorAll('.alert').length,
            hud: document.getElementById('hud-count').textContent,
            level: document.getElementById('hud-level').textContent,
            overlay: [document.getElementById('overlay').width, document.getElementById('overlay').height],
            trend: [document.getElementById('trend').width, document.getElementById('trend').height],
            feedOk: (() => { const i = document.getElementById('feed'); return !!(i && i.complete && i.naturalWidth > 0); })(),
            engine: document.getElementById('engine-state').textContent,
        }))()"""
        res = await call("Runtime.evaluate", {"expression": expr, "returnByValue": True})
        facts = res.get("result", {}).get("result", {}).get("value")
        if facts is None:
            print("RAW EVAL:", json.dumps(res)[:800])
            url_res = await call("Runtime.evaluate", {"expression": "location.href + ' | ' + document.readyState", "returnByValue": True})
            print("PAGE:", json.dumps(url_res)[:400])

        if pre_shot_js:
            await call("Runtime.evaluate", {"expression": pre_shot_js, "returnByValue": True})
            await asyncio.sleep(3)

        data = None
        for attempt in range(3):
            try:
                shot = await call("Page.captureScreenshot",
                                  {"format": "png", "captureBeyondViewport": False,
                                   "optimizeForSpeed": True}, timeout=60)
                data = shot.get("result", {}).get("data")
                if data:
                    break
            except Exception as exc:  # noqa: BLE001 - retry a busy renderer
                print(f"screenshot attempt {attempt + 1} failed: {exc!r}")
                await asyncio.sleep(2)
        if data:
            with open(screenshot_path, "wb") as fh:
                fh.write(base64.b64decode(data))

    problems = []
    for ev in events:
        m = ev.get("method")
        if m == "Runtime.exceptionThrown":
            details = ev["params"]["exceptionDetails"]
            problems.append("EXCEPTION: " + str(details.get("text")) + " " +
                            str(details.get("exception", {}).get("description", ""))[:300])
        elif m == "Log.entryAdded":
            entry = ev["params"]["entry"]
            if entry.get("level") in ("error", "warning"):
                problems.append(f"{entry['level'].upper()}: {entry.get('text')} {entry.get('url','')}")
        elif m == "Runtime.consoleAPICalled" and ev["params"].get("type") == "error":
            args = [str(a.get("value", a.get("description", ""))) for a in ev["params"].get("args", [])]
            problems.append("CONSOLE: " + " ".join(args)[:300])

    return facts, problems


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/"
    out = sys.argv[2] if len(sys.argv) > 2 else "ui_check.png"

    profile = tempfile.mkdtemp(prefix="gdv-chrome-")
    proc = subprocess.Popen(
        [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
         "--mute-audio", f"--remote-debugging-port={PORT}",
         "--window-size=1680,1050", f"--user-data-dir={profile}", url],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        targets = None
        for _ in range(80):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json", timeout=2) as r:
                    targets = json.load(r)
                if targets:
                    break
            except Exception:
                pass
            time.sleep(0.4)
        if not targets:
            print("FAILED: chrome devtools never came up")
            return 1
        # Create a dedicated tab so we never attach to the wrong target.
        page = None
        try:
            new_url = (f"http://127.0.0.1:{PORT}/json/new?"
                       + urllib.parse.quote(url, safe=""))
            req = urllib.request.Request(new_url, method="PUT")
            with urllib.request.urlopen(req, timeout=6) as r:
                page = json.load(r)
        except Exception as exc:  # noqa: BLE001 - fall back to an existing tab
            print("target create warning:", repr(exc))
        if not page or not page.get("webSocketDebuggerUrl"):
            page = next((t for t in targets if t.get("type") == "page"), targets[0])
        pre_js = sys.argv[3] if len(sys.argv) > 3 else None
        facts, problems = asyncio.run(drive(page["webSocketDebuggerUrl"], out, pre_js, url))
        print("DOM:", json.dumps(facts, indent=2))
        print("PROBLEMS:", "none" if not problems else "")
        for p in problems:
            print("  -", p)
        print("SCREENSHOT:", out, os.path.getsize(out) if os.path.exists(out) else "missing")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
