#!/usr/bin/env python3
"""
YOLO PT 모델 노드 (탐사 노드 연동)
- PT 모델(.pt)을 사용
- /yolo_detections (JSON) 토픽 발행하여 탐사 노드와 연동

실행: python3 yolo_test1.py
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import time
import numpy as np
import json
import math

# ================= [설정] =================
MODEL_PATH = "yolov11_best.pt"
CAMERA_TOPIC = "/oakd/rgb/preview/image_raw/compressed"

# 목표 FPS (10~15 사이 추천)
TARGET_FPS = 15

# 이미지 크기
IMG_SIZE = 320

# 카메라 수평 시야각 (deg) - OAK-D 기본값 기준
CAMERA_HFOV_DEG = 69.0

# 신뢰도 임계값
CONF_THRESHOLD = 0.75

# 박스 좌표 평균화 설정
BOX_BUFFER_SIZE = 1  # 몇 개 모아서 평균 낼지 (15 -> 5로 축소: 로봇 이동 중 오차 감소)
# ==========================================


class Yolo11Node(Node):
    def __init__(self):
        super().__init__('yolo11_node')
        
        # QoS: 최신 프레임만
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # 카메라 구독
        self.sub = self.create_subscription(
            CompressedImage, CAMERA_TOPIC, self.image_callback, qos
        )
        
        # 결과 발행 (이미지)
        self.pub = self.create_publisher(CompressedImage, '/yolo_result', 10)
        
        # 감지 결과 발행 (JSON) - 다른 노드에서 구독 가능
        self.det_pub = self.create_publisher(String, '/yolo_detections', 10)
        
        self.bridge = CvBridge()
        self.latest_frame = None
        
        # YOLO 모델 로드
        self.get_logger().info(f"모델 로딩: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)
        
        # 워밍업
        self.get_logger().info("워밍업...")
        dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        self.model(dummy, imgsz=IMG_SIZE, verbose=False)
        self.get_logger().info(f"준비 완료! 목표 FPS: {TARGET_FPS}")
        
        # 상태
        self.last_boxes = []
        self.last_inference_ms = 0.0
        self.actual_fps = 0.0
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # 박스 좌표 버퍼 (평균화용)
        self.box_buffer = []  # [{angle_rad, conf, center}, ...]
        self.no_box_count = 0  # 박스 미감지 연속 횟수
        
        # 타이머: 고정 FPS로 처리
        timer_period = 1.0 / TARGET_FPS
        self.timer = self.create_timer(timer_period, self.process_frame)

    def image_callback(self, msg):
        """카메라 프레임 저장 (최신 것만)"""
        try:
            self.latest_frame = self.bridge.compressed_imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().error(f"이미지 변환 에러: {e}")

    def process_frame(self):
        """타이머로 호출 - 고정 FPS 처리"""
        if self.latest_frame is None:
            return
        
        frame = self.latest_frame.copy()
        frame_h, frame_w = frame.shape[:2]
        self.frame_count += 1
        
        try:
            # YOLO 추론
            t_start = time.time()
            results = self.model(
                frame, 
                imgsz=IMG_SIZE, 
                conf=CONF_THRESHOLD, 
                verbose=False
            )
            self.last_inference_ms = (time.time() - t_start) * 1000
            
            # 결과 파싱
            self.last_boxes = []
            half_fov_rad = math.radians(CAMERA_HFOV_DEG) / 2.0
            half_frame_w = frame_w / 2.0 if frame_w else 1.0

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = self.model.names.get(cls_id, f"ID:{cls_id}")
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    bearing_ratio = (cx - half_frame_w) / half_frame_w
                    angle_rad = max(-half_fov_rad,
                                    min(half_fov_rad, bearing_ratio * half_fov_rad))
                    
                    self.last_boxes.append({
                        'box': [x1, y1, x2, y2],
                        'center': [cx, cy],  # 중심점 추가
                        'label': f"{name} {conf:.2f}",
                        'name': name,
                        'conf': conf,
                        'angle_rad': angle_rad,
                        'angle_deg': math.degrees(angle_rad)
                    })
            
            # 박스 감지 여부 확인
            box_detections = [item for item in self.last_boxes if item['name'].lower() == 'box']
            
            # 박스 좌표 버퍼링 및 평균화 발행
            if box_detections:
                self.no_box_count = 0  # 리셋
                
                # 가장 신뢰도 높은 박스 선택
                best_box = max(box_detections, key=lambda x: x['conf'])
                
                # 버퍼에 추가
                self.box_buffer.append({
                    'angle_rad': best_box['angle_rad'],
                    'angle_deg': best_box['angle_deg'],
                    'conf': best_box['conf'],
                    'center': best_box['center'],
                    'box': [x1, y1, x2, y2],
                })
                
                # 버퍼가 다 차면 평균 계산 후 발행
                if len(self.box_buffer) >= BOX_BUFFER_SIZE:
                    avg_angle_rad = sum(b['angle_rad'] for b in self.box_buffer) / len(self.box_buffer)
                    avg_angle_deg = sum(b['angle_deg'] for b in self.box_buffer) / len(self.box_buffer)
                    avg_conf = sum(b['conf'] for b in self.box_buffer) / len(self.box_buffer)
                    avg_cx = sum(b['center'][0] for b in self.box_buffer) / len(self.box_buffer)
                    avg_cy = sum(b['center'][1] for b in self.box_buffer) / len(self.box_buffer)
                    avg_box_x1 = sum(b['box'][0] for b in self.box_buffer) / len(self.box_buffer)
                    avg_box_y1 = sum(b['box'][1] for b in self.box_buffer) / len(self.box_buffer)
                    avg_box_x2 = sum(b['box'][2] for b in self.box_buffer) / len(self.box_buffer)
                    avg_box_y2 = sum(b['box'][3] for b in self.box_buffer) / len(self.box_buffer)
                    
                    # 평균 좌표로 토픽 발행
                    avg_box = {
                        'name': 'box',
                        'angle_rad': avg_angle_rad,
                        'angle_deg': avg_angle_deg,
                        'conf': avg_conf,
                        'center': [avg_cx, avg_cy],
                        'box': [avg_box_x1, avg_box_y1, avg_box_x2, avg_box_y2]
                    }
                    
                    det_msg = String()
                    det_msg.data = json.dumps([avg_box])
                    self.det_pub.publish(det_msg)
                    
                    self.get_logger().warn(
                        f"🎯 박스 평균 좌표 발행! "
                        f"각도={avg_angle_deg:.1f}deg, conf={avg_conf:.2f}, "
                        f"중심=({avg_cx:.0f}, {avg_cy:.0f})"
                    )
                    
                    # 버퍼 초기화 → 다시 모으기 시작
                    self.box_buffer = []
            else:
                # 박스 미감지 시 카운트 증가
                self.no_box_count += 1
                # 5프레임 이상 미감지면 버퍼 초기화 (새 박스 준비)
                if self.no_box_count >= 5 and self.box_buffer:
                    self.get_logger().info("박스 미감지 - 버퍼 초기화")
                    self.box_buffer = []
            
            # 시각화
            for item in self.last_boxes:
                x1, y1, x2, y2 = item['box']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = item['label']
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # FPS 계산 (1초마다 갱신)
            elapsed = time.time() - self.fps_start_time
            if elapsed >= 1.0:
                self.actual_fps = self.frame_count / elapsed
                self.frame_count = 0
                self.fps_start_time = time.time()
            
            # 상태 표시
            info = f"FPS:{self.actual_fps:.1f} | Inf:{self.last_inference_ms:.0f}ms | Obj:{len(self.last_boxes)}"
            cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 발행
            out_msg = self.bridge.cv2_to_compressed_imgmsg(frame)
            self.pub.publish(out_msg)
            
        except Exception as e:
            self.get_logger().error(f"처리 에러: {e}")


def main():
    rclpy.init()
    
    print("=" * 50)
    print(f"  YOLO PT 모드 (탐사 연동)")
    print("=" * 50)
    print(f"  모델: {MODEL_PATH}")
    print(f"  JSON 토픽: /yolo_detections")
    print("=" * 50)
    
    try:
        node = Yolo11Node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
