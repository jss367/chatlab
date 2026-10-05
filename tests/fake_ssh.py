"""Local stand-in for an owned SSH master and its forwarding control request.

The master executes the remote command here. Its Unix control socket accepts
only the generated forward and serves it until the command exits. No real
SSH host or user configuration is accessed.
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time


def _relay(source, sink):
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


def _serve_forward(listener, target_host, target_port):
    while True:
        try:
            client, _ = listener.accept()
        except OSError:
            return
        try:
            upstream = socket.create_connection((target_host, int(target_port)))
        except OSError:
            client.close()
            continue
        threading.Thread(target=_relay, args=(client, upstream), daemon=True).start()
        threading.Thread(target=_relay, args=(upstream, client), daemon=True).start()


def _serve_control(listener):
    while True:
        try:
            client, _ = listener.accept()
        except OSError:
            return
        with client, client.makefile("r") as stream:
            bind_host, bind_port, target_host, target_port = json.loads(stream.readline()).split(":")
            forward = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                forward.bind((bind_host, int(bind_port)))
                forward.listen()
            except OSError as error:
                forward.close()
                result = {"status": 255, "error": f"bind [{bind_host}]:{bind_port}: {error}\nCould not request local forwarding."}
            else:
                threading.Thread(target=_serve_forward, args=(forward, target_host, target_port), daemon=True).start()
                result = {"status": 0, "error": ""}
            client.sendall((json.dumps(result) + "\n").encode())


def main(arguments):
    forward = control_path = operation = None
    positional = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in ("-o", "-L", "-S", "-O", "-F"):
            value = arguments[index + 1]
            if argument == "-L":
                forward = value
            elif argument == "-S":
                control_path = value
            elif argument == "-O":
                operation = value
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
    if operation:
        time.sleep(float(os.environ.get("FAKE_SSH_FORWARD_DELAY", "0")))
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(control_path)
            client.sendall((json.dumps(forward) + "\n").encode())
            with client.makefile("r") as stream:
                result = json.loads(stream.readline())
            if result["error"]:
                print(result["error"], file=sys.stderr)
            sys.exit(result["status"])
    _host, *command = positional
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(control_path)
        listener.listen()
        threading.Thread(target=_serve_control, args=(listener,), daemon=True).start()
        if os.environ.get("FAKE_SSH_VERBOSE_TUNNEL"):
            for index in range(1024):
                print(f"diagnostic {index}: " + "x" * 512, file=sys.stderr)
            sys.stderr.flush()
        sys.exit(subprocess.call(["sh", "-c", " ".join(command)]))


if __name__ == "__main__":
    main(sys.argv[1:])
