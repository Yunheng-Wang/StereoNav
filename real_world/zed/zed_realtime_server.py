import sys
import json
import os
import pyzed.sl as sl
import cv2
import time
import threading
import argparse
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer

_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zed_config.json")
with open(_cfg_path) as _f:
    _cfg = json.load(_f)

sys.path.insert(0, _cfg["unitree_sdk_path"])
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

HOST         = _cfg["host"]
PORT         = _cfg["port"]
WIDTH        = _cfg["width"]
HEIGHT       = _cfg["height"]
JPEG_QUALITY = _cfg["jpeg_quality"]
HFOV         = _cfg["hfov"]
CAM_HEIGHT   = _cfg["cam_height"]
CAM_BASELINE = _cfg["cam_baseline"]


def get_K():
    hfov_rad = np.deg2rad(HFOV)
    fx = WIDTH / (2.0 * np.tan(hfov_rad / 2.0))
    return np.array([[fx, 0, WIDTH/2.0], [0, fx, HEIGHT/2.0], [0, 0, 1]], dtype=np.float64)

K = get_K()


# ---------- 里程计状态（线程安全） ----------
class OdomState:
    def __init__(self):
        self.lock = threading.Lock()
        # 世界坐标系下机身几何中心位置 [x, y, z]
        self.position = np.zeros(3)
        # 四元数 [w, x, y, z]（机器人坐标系相对世界坐标系的旋转）
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.ready = False

odom_state = OdomState()


def odom_callback(msg: SportModeState_):
    pos = np.array(msg.position, dtype=np.float64)       # [x, y, z] 世界系
    q = np.array(msg.imu_state.quaternion, dtype=np.float64)  # [w, x, y, z]
    with odom_state.lock:
        odom_state.position = pos
        odom_state.quat = q
        odom_state.ready = True


def quat_to_rot(q):
    """四元数 [w, x, y, z] -> 3x3 旋转矩阵 R（世界->机器人 body）"""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def world_to_image(target_world, cam_side):
    with odom_state.lock:
        pos = odom_state.position.copy()
        q = odom_state.quat.copy()

    # quat_to_rot 通常是 body -> world
    R_body_to_world = quat_to_rot(q)
    R_world_to_body = R_body_to_world.T

    if cam_side == 'left':
        cam_offset_body = np.array([0.0,  CAM_BASELINE / 2.0, CAM_HEIGHT])
    else:
        cam_offset_body = np.array([0.0, -CAM_BASELINE / 2.0, CAM_HEIGHT])

    # body offset -> world
    cam_pos_world = pos + R_body_to_world @ cam_offset_body

    # target relative to camera in world
    rel_world = target_world - cam_pos_world

    # world -> body
    rel_body = R_world_to_body @ rel_world

    # G1 body: x前, y左, z上
    # CV camera: x右, y下, z前
    cam_x = -rel_body[1]
    cam_y = -rel_body[2]
    cam_z =  rel_body[0]

    if cam_z <= 0:
        u = 0.0 if cam_x < 0 else float(WIDTH - 1)
        v = HEIGHT / 2.0
        return u, v

    u = K[0, 0] * cam_x / cam_z + K[0, 2]
    v = K[1, 1] * cam_y / cam_z + K[1, 2]
    u = float(np.clip(u, 0, WIDTH - 1))
    v = float(np.clip(v, 0, HEIGHT - 1))
    return u, v

def draw_red_dot(bgr, u, v, radius=8):
    cv2.circle(bgr, (int(round(u)), int(round(v))), radius, (0, 0, 255), -1)
    return bgr


def encode_jpg(image_bgr):
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return buf.tobytes() if ok else None


# ---------- 帧缓冲 ----------
class FrameBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.left_jpg = None
        self.right_jpg = None
        self.frame_id = 0
        self.timestamp = 0.0
        self.last_error = ""

frame_buffer = FrameBuffer()

# 目标点（世界坐标系，由命令行参数设置）
target_world = np.array([0.0, 0.0, 0.0])


def zed_capture_loop():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.VGA
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NONE

    print("Opening ZED Mini...")
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED:", err)
        with frame_buffer.lock:
            frame_buffer.last_error = str(err)
        return

    print("ZED opened successfully.")
    left_image = sl.Mat()
    right_image = sl.Mat()

    for _ in range(30):
        zed.grab()
        time.sleep(0.01)

    while True:
        err = zed.grab()
        if err == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            zed.retrieve_image(right_image, sl.VIEW.RIGHT)

            left_bgr = cv2.cvtColor(left_image.get_data(), cv2.COLOR_BGRA2BGR)
            right_bgr = cv2.cvtColor(right_image.get_data(), cv2.COLOR_BGRA2BGR)
            left_bgr = cv2.resize(left_bgr, (WIDTH, HEIGHT))
            right_bgr = cv2.resize(right_bgr, (WIDTH, HEIGHT))

            if not odom_state.ready:
                continue

            tgt = target_world.copy()
            u_l, v_l = world_to_image(tgt, 'left')
            u_r, v_r = world_to_image(tgt, 'right')
            left_bgr = draw_red_dot(left_bgr, u_l, v_l)
            right_bgr = draw_red_dot(right_bgr, u_r, v_r)

            left_jpg = encode_jpg(left_bgr)
            right_jpg = encode_jpg(right_bgr)

            if left_jpg and right_jpg:
                with frame_buffer.lock:
                    frame_buffer.left_jpg = left_jpg
                    frame_buffer.right_jpg = right_jpg
                    frame_buffer.frame_id += 1
                    frame_buffer.timestamp = time.time()
                    frame_buffer.last_error = ""
        else:
            with frame_buffer.lock:
                frame_buffer.last_error = str(err)

        time.sleep(0.001)


# ---------- HTTP 服务 ----------
class ZEDRequestHandler(BaseHTTPRequestHandler):
    def send_jpg(self, jpg_bytes):
        if jpg_bytes is None:
            self.send_response(503)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"No frame available yet.")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpg_bytes)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.end_headers()
        self.wfile.write(jpg_bytes)

    def do_GET(self):
        with frame_buffer.lock:
            left_jpg = frame_buffer.left_jpg
            right_jpg = frame_buffer.right_jpg
            frame_id = frame_buffer.frame_id
            timestamp = frame_buffer.timestamp
            last_error = frame_buffer.last_error
        with odom_state.lock:
            pos = odom_state.position.tolist()
            q = odom_state.quat.tolist()

        if self.path == "/left.jpg":
            self.send_jpg(left_jpg)
        elif self.path == "/right.jpg":
            self.send_jpg(right_jpg)
        elif self.path == "/status":
            msg = (
                f"frame_id: {frame_id}\n"
                f"timestamp: {timestamp}\n"
                f"last_error: {last_error}\n"
                f"odom_position(x_fwd,y_left,z_up): {pos}\n"
                f"odom_quat(w,x,y,z): {q}\n"
                f"target_world(x,y,z): {target_world.tolist()}\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(msg.encode("utf-8"))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ZED realtime server.\n/left.jpg /right.jpg /status\n")

    def do_POST(self):
        if self.path == "/target":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)

                # 1. 更新目标点
                target_world[:] = [
                    float(data["x"]),
                    float(data["y"]),
                    float(data.get("z", CAM_HEIGHT))
                ]
                print(f"Target updated: {target_world.tolist()}")

                # 2. 只在 POST /target 时等 2 秒
                time.sleep(2.0)

                # 3. 再返回响应
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "target": target_world.tolist(),
                    "message": "target updated, waited 1s for frame refresh"
                }).encode())

            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    
    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, default=_cfg.get("network", "eth0"),
                        help="DDS 网络接口名，默认读 config.json")
    args = parser.parse_args()

    print("Target not set yet. Use POST /target {\"x\":...,\"y\":...,\"z\":...} to set.")

    # 初始化 DDS，订阅里程计
    ChannelFactoryInitialize(0, args.network)
    odom_sub = ChannelSubscriber("rt/odommodestate", SportModeState_)
    odom_sub.Init(odom_callback, 10)
    print(f"Subscribed to rt/odommodestate on {args.network}")

    capture_thread = threading.Thread(target=zed_capture_loop, daemon=True)
    capture_thread.start()

    server = HTTPServer((HOST, PORT), ZEDRequestHandler)
    print(f"ZED realtime server started at http://{HOST}:{PORT}")
    server.serve_forever()


# g1 上执行示例：
# /usr/bin/python3 zed_realtime_server.py --network eth0
