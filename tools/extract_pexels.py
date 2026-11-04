"""Extract direct Pexels video file URLs from a saved search page."""
import re
import sys

html = open("pexels.html", encoding="utf-8", errors="ignore").read()
plain = html.replace("\\u002F", "/")
urls = sorted(set(
    re.findall(r"videos\.pexels\.com/video-files/[0-9]+/[0-9A-Za-z_.\-]+\.mp4", plain)
))
for u in urls:
    print(u)
open("pexels_urls.txt", "w").write("\n".join(urls))
print(f"-- {len(urls)} urls")
