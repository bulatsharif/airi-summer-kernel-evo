from __future__ import annotations

import argparse
import os
import select
import socket
import threading

import paramiko


def relay(client_socket: socket.socket, channel: paramiko.Channel) -> None:
    try:
        while True:
            readable, _, _ = select.select([client_socket, channel], [], [])
            if client_socket in readable:
                data = client_socket.recv(65536)
                if not data:
                    break
                channel.sendall(data)
            if channel in readable:
                data = channel.recv(65536)
                if not data:
                    break
                client_socket.sendall(data)
    finally:
        channel.close()
        client_socket.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--local-port", type=int, required=True)
    parser.add_argument("--remote-port", type=int, required=True)
    args = parser.parse_args()

    password = os.environ.get("SCHOOL17_PASSWORD")
    if not password:
        raise RuntimeError("SCHOOL17_PASSWORD is not set")

    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.load_host_keys(os.path.expanduser("~/.ssh/known_hosts"))
    ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
    ssh.connect(
        hostname=args.host,
        username=args.user,
        password=password,
        allow_agent=False,
        look_for_keys=False,
        timeout=15,
        auth_timeout=15,
    )

    transport = ssh.get_transport()
    if transport is None:
        raise RuntimeError("SSH transport was not created")
    transport.set_keepalive(30)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", args.local_port))
    listener.listen(32)

    try:
        while transport.is_active():
            client_socket, client_address = listener.accept()
            try:
                channel = transport.open_channel(
                    "direct-tcpip",
                    ("127.0.0.1", args.remote_port),
                    client_address,
                )
            except Exception:
                client_socket.close()
                continue
            threading.Thread(
                target=relay,
                args=(client_socket, channel),
                daemon=True,
            ).start()
    finally:
        listener.close()
        ssh.close()


if __name__ == "__main__":
    main()
