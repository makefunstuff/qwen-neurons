"""python -m qwen_neurons [--no-open]: start the app.

A browser tab opens only when started from an interactive terminal, and never with --no-open (or --no-browser)
or QN_NO_BROWSER=1."""
import sys
import threading
import time
import webbrowser

import uvicorn

from .config import Settings
from .server import create_app


def should_open(s: Settings, argv: list[str]) -> bool:
    if s.no_browser or "--no-open" in argv or "--no-browser" in argv:
        return False
    return sys.stdin.isatty() and sys.stdout.isatty()


def main() -> None:
    s = Settings.load()
    shown = "localhost" if s.host in ("127.0.0.1", "0.0.0.0", "::") else s.host
    url = f"http://{shown}:{s.port}/"
    print(f"\n  😈🐶 Evil Pugs → {url}\n     model: {s.model}   device: {s.device}   idle unload: {s.idle_timeout_s}s\n",
          flush=True)
    if should_open(s, sys.argv[1:]):
        threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open(url)), daemon=True).start()
    uvicorn.run(create_app(s), host=s.host, port=s.port, log_level="warning")


if __name__ == "__main__":
    main()
