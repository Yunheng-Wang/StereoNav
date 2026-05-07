"""
action_server.py — 运行在 G1 上，接收外部电脑发来的动作列表并执行。

外部电脑调用示例：
    curl -X POST http://<G1_IP>:8001/execute \
         -H "Content-Type: application/json" \
         -d '{"actions": [1, 1, 1, 1]}'

动作定义：
    0: stop
    1: move forward
    2: turn left
    3: turn right
"""
import sys
import json
import os
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "action_config.json")
with open(_cfg_path) as _f:
    _cfg = json.load(_f)

sys.path.insert(0, _cfg["unitree_sdk_path"])
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

ACTION_PORT   = _cfg.get("action_port", 8001)
MOVE_SPEED    = _cfg.get("move_speed", 0.25)
TURN_SPEED    = _cfg.get("turn_speed", 0.5)
MOVE_DURATION = _cfg.get("move_duration", 1.0)
TURN_DURATION = _cfg.get("turn_duration", 0.5236)

loco: LocoClient = None
exec_lock = threading.Lock()


def compress_actions(actions):
    if not actions:
        return []
    compressed = []
    last = actions[0]
    count = 1
    for a in actions[1:]:
        if a == last:
            count += 1
        else:
            compressed.append((last, count))
            last = a
            count = 1
    compressed.append((last, count))
    return compressed


def send_move_for_duration(vx, vy, wz, duration, hz=20):
    dt = 1.0 / hz
    start = time.time()
    while time.time() - start < duration:
        loco.Move(vx, vy, wz)
        time.sleep(dt)
    loco.Move(0, 0, 0)
    time.sleep(0.1)


def execute_actions(actions):
    with exec_lock:
        compressed = compress_actions(actions)
        print(f"Received actions: {actions}")
        print(f"Compressed actions: {compressed}")

        for action, count in compressed:
            if action == 0:
                print("Stop")
                loco.Move(0, 0, 0)
                time.sleep(0.1)
            elif action == 1:
                duration = MOVE_DURATION * count
                print(f"Move forward for {duration:.2f}s")
                send_move_for_duration(MOVE_SPEED, 0, 0, duration)
            elif action == 2:
                duration = TURN_DURATION * count
                print(f"Turn left for {duration:.2f}s")
                send_move_for_duration(0, 0, TURN_SPEED, duration)
            elif action == 3:
                duration = TURN_DURATION * count
                print(f"Turn right for {duration:.2f}s")
                send_move_for_duration(0, 0, -TURN_SPEED, duration)
            else:
                print(f"Unknown action: {action}, skip.")


class ActionHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/execute":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            data = json.loads(body)
            actions = data["actions"]
            assert isinstance(actions, list)
        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        # 关键修改：
        # 不再开后台线程，而是在当前 HTTP 请求里同步执行
        # 这样只有 execute_actions(actions) 完成后，才会返回响应
        try:
            execute_actions(actions)
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "error",
                "error": str(e),
                "actions": actions,
            }).encode())
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "status": "done",
            "actions": actions,
        }).encode())

    def do_GET(self):
        if self.path == "/status":
            busy = exec_lock.locked()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"busy": busy}).encode())
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Action server. POST /execute {\"actions\": [1,1,0]}\n")

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, default=_cfg.get("network", "eth0"))
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.network)
    loco = LocoClient()
    loco.SetTimeout(10.0)
    loco.Init()
    print("G1 LocoClient initialized.")

    server = HTTPServer(("0.0.0.0", ACTION_PORT), ActionHandler)
    print(f"Action server started at http://0.0.0.0:{ACTION_PORT}")
    server.serve_forever()
