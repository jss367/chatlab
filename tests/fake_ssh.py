"""Stands in for ``ssh`` in the remote session tests.

It ignores the host. Given a command, it runs it here through ``sh``, the
way sshd would run it on the far side, and exits when it does. Given
``-L``, it also listens on the local end and relays each connection to the
target itself, the way a real forward would, and exits with SSH's status
when it cannot listen.
"""

import os
import socket
import subprocess
import sys
import threading

_listening = threading.Event()


def _relay(source: socket.socket, sink: socket.socket) -> None:
    try:
        while data := source.recv(65536):
            sink.sendall(data)
    except OSError:
        pass
    finally:
        try:
            sink.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _forward(spec: str) -> None:
    bind_host, bind_port, target_host, target_port = spec.split(":")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind((bind_host, int(bind_port)))
    except OSError as error:
        print(f"bind [{bind_host}]:{bind_port}: {error.strerror}", file=sys.stderr)
        print("Could not request local forwarding.", file=sys.stderr)
        os._exit(255)
    listener.listen()
    _listening.set()
    while True:
        client, _ = listener.accept()
        try:
            upstream = socket.create_connection((target_host, int(target_port)))
        except OSError:
            client.close()
            continue
        threading.Thread(target=_relay, args=(client, upstream), daemon=True).start()
        threading.Thread(target=_relay, args=(upstream, client), daemon=True).start()


def main(arguments: list[str]) -> None:
    forward = None
    positional = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in ("-o", "-L"):
            if argument == "-L":
                forward = arguments[index + 1]
            index += 2
        elif argument.startswith("-"):
            index += 1
        else:
            positional.append(argument)
            index += 1
    refusal = os.environ.get("FAKE_SSH_REFUSE")
    if refusal:
        print(refusal, file=sys.stderr)
        sys.exit(255)
    _host, *command = positional
    if forward is not None:
        if os.environ.get("FAKE_SSH_VERBOSE_TUNNEL"):
            for index in range(1024):
                print(f"diagnostic {index}: " + "x" * 512, file=sys.stderr)
            sys.stderr.flush()
        threading.Thread(target=_forward, args=(forward,), daemon=True).start()
        if not _listening.wait(5):
            sys.exit(255)
        if not command:
            threading.Event().wait()
        sys.exit(subprocess.call(["sh", "-c", " ".join(command)]))
    os.execvp("sh", ["sh", "-c", " ".join(command)])


if __name__ == "__main__":
    main(sys.argv[1:])
