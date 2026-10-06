"""
Connect 4 AlphaGo Studio Launcher
Run this script to launch the local web server and automatically open the analysis studio in your browser.
Usage:
    python run_studio.py [--port 8000] [--no-browser]
"""
import argparse
import threading
import webbrowser
import uvicorn
from web_server import app


def main():
    parser = argparse.ArgumentParser(description="Connect 4 AlphaGo Studio")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()

    print("=======================================================")
    print(f" Connect 4 AlphaGo Studio running at http://{args.host}:{args.port}")
    print(" Press Ctrl+C to stop.")
    print("=======================================================")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
