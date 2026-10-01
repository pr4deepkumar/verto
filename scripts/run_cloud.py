"""
scripts/run_cloud.py
~~~~~~~~~~~~~~~~~~~~
Launcher script for VERTO Dashboard & API server.

Starts the FastAPI server on 0.0.0.0 and prints the command that points the CLI at it.

Note on --ngrok:
    --ngrok is OPTIONAL. It is only required if you choose to serve inference
    live directly inside a Google Colab notebook cell without downloading `best_model.pt`.
    If running locally or deploying on AWS, ngrok is not required.

Usage:
    python scripts/run_cloud.py
    python scripts/run_cloud.py --port 8000
    python scripts/run_cloud.py --ngrok  (Optional: Live Colab serving)
"""

import argparse
import os
import socket
import sys

import uvicorn

from verto import config as cfg


def get_lan_ip() -> str:
    """This machine's address on its local network (not its public IP), best effort."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # UDP: picks the outgoing interface without sending anything
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main():
    parser = argparse.ArgumentParser(description="Run VERTO Cloud Dashboard & API service.")
    parser.add_argument("--host", default=cfg.HOST, help="Host IP binding.")
    parser.add_argument("--port", type=int, default=cfg.PORT, help="Port number.")
    parser.add_argument("--ngrok", action="store_true", help="Optional: Establish a public HTTPS ngrok tunnel (only for live Colab serving).")
    parser.add_argument("--auth-token", default=os.environ.get("NGROK_AUTHTOKEN"), help="Ngrok authtoken.")
    args = parser.parse_args()

    server_url = f"http://{get_lan_ip()}:{args.port}"

    print("\n" + "=" * 64)
    print(" VERTO Cloud Dashboard Launcher")
    print("=" * 64)
    print(f" Binding Host : {args.host}")
    print(f" Port         : {args.port}")

    # The engine needs the Spider databases locally to execute queries. Check up
    # front so we fail with a clear message instead of a cryptic startup crash.
    tables_json = os.path.join(cfg.SPIDER_DIR, "tables.json")
    if not os.path.exists(tables_json):
        print(f"\n [Error] Spider dataset not found (missing '{tables_json}').")
        print("         The server needs the databases locally to run queries.")
        print("         Run this once:  python scripts/setup_data.py\n")
        sys.exit(1)

    checkpoint_path = cfg.CHECKPOINT_PATH
    if not os.path.exists(checkpoint_path):
        print(f" [Notice] Model checkpoint not found at '{checkpoint_path}'.")
        print("          If trained on Google Colab, copy `best_model.pt` into `./checkpoints/`.")

    if args.ngrok:
        try:
            from pyngrok import ngrok  # type: ignore
            if args.auth_token:
                ngrok.set_auth_token(args.auth_token)
            tunnel = ngrok.connect(args.port)
            server_url = tunnel.public_url
            print(f" Ngrok Tunnel : {server_url}")
        except ImportError:
            print(" [Warning] pyngrok not installed. Run `pip install -e \".[ngrok]\"` for automated tunneling.")
        except Exception as e:
            print(f" [Warning] Failed to launch ngrok tunnel: {e}")

    print("\n [Local CLI Connection Command]")
    print(f"   export VERTO_SERVER_URL={server_url}")
    if not args.ngrok:
        print("   (a local-network address; from outside, use the machine's public IP or --ngrok)")
    print("   verto --db concert_singer \"How many singers are older than 30?\"")
    print("=" * 64 + "\n")

    uvicorn.run("verto.api.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
