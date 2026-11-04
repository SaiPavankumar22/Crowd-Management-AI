"""Download a sample crowd video into samples/ (fallback used when a file
argument is given with -i)."""
import sys
import urllib.request

url = sys.argv[1]
out = sys.argv[2]
req = urllib.request.Request(url, headers={
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126 Safari/537.36"})
with urllib.request.urlopen(req, timeout=180) as resp, open(out, "wb") as fh:
    total = 0
    while True:
        chunk = resp.read(1 << 20)
        if not chunk:
            break
        fh.write(chunk)
        total += len(chunk)
print(f"downloaded {total} bytes -> {out}")
