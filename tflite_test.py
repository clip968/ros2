#!/usr/bin/env python3
"""
YOLO11n TFLite 테스트 노드 (yolo_test1.py 호환 버전)
- TFLite 모델(.tflite)을 사용하여 속도 향상
- /yolo_detections (JSON) 토픽 발행하여 탐사 노드와 연동
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import time
import numpy as np
import json

# ================= [설정] =================
MODEL_PATH = "best_int8.tflite"   # TFLite 모델 경로
CAMERA_TOPIC = "/oakd/rgb/preview/image_raw"
TARGET_FPS = 15
IMG_SIZE =320         # 모델 변환 시 크기와 동일해야 함
CONF_THRESHOLD = 0.4
# ==========================================

class YoloTfliteNode(Node):
    def __init__(self):
        super().__init__('yolo_tflite_node')
        
        # QoS 설정
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # 구독자
        self.sub = self.create_subscription(
            Image, CAMERA_TOPIC, self.image_callback, qos
        )
        
        # 발행자
        self.pub = self.create_publisher(Image, '/yolo_result', 10)
        self.det_pub = self.create_publisher(String, '/yolo_detections', 10)  # JSON 발행
        
        self.bridge = CvBridge()
        self.latest_frame = None
        
        # 모델 로드
        self.get_logger().info(f"⚡ TFLite 모델 로딩: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH, task='detect')
        
        # 워밍업
        self.get_logger().info("🔥 워밍업...")
        dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        self.model(dummy, imgsz=IMG_SIZE, verbose=False)
        self.get_logger().info(f"✅ 준비 완료! 목표 FPS: {TARGET_FPS}")
        
        # 상태 변수
        self.last_boxes = []
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.actual_fps = 0.0
        
        # 타이머
        timer_period = 1.0 / TARGET_FPS
        self.timer = self.create_timer(timer_period, self.process_frame)

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"이미지 변환 에러: {e}")

    def process_frame(self):
        if self.latest_frame is None:
            return
        
        frame = self.latest_frame.copy()
        self.frame_count += 1
        
        try:
            t_start = time.time()
            
            # TFLite 추론
            results = self.model(
                frame, 
                imgsz=IMG_SIZE, 
                conf=CONF_THRESHOLD, 
                verbose=False
            )
            
            inference_ms = (time.time() - t_start) * 1000
            
            # 결과 파싱 및 JSON 발행
            self.last_boxes = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = self.model.names.get(cls_id, f"ID:{cls_id}")
                    
                    self.last_boxes.append({
                        'box': [x1, y1, x2, y2],
                        'center': [(x1+x2)/2, (y1+y2)/2],  # 중심점 추가
                        'label': f"{name} {conf:.2f}",
                        'name': name,
                        'conf': conf
                    })
            
            # JSON 발행 (탐사 노드용)
            if self.last_boxes:
                det_msg = String()
                det_msg.data = json.dumps(self.last_boxes)
                self.det_pub.publish(det_msg)
                
                # 로그 출력 (간단하게)
                names = [b['name'] for b in self.last_boxes]
                # self.get_logger().info(f"🔍 감지: {names}")

            # 시각화 (yolo_test1.py 스타일)
            for item in self.last_boxes:
                x1, y1, x2, y2 = item['box']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = item['label']
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # FPS 계산
            elapsed = time.time() - self.fps_start_time
            if elapsed >= 1.0:
                self.actual_fps = self.frame_count / elapsed
                self.frame_count = 0
                self.fps_start_time = time.time()
            
            # 정보 표시
            info = f"TFLite FPS:{self.actual_fps:.1f} | {inference_ms:.0f}ms | Obj:{len(self.last_boxes)}"
            cv2.putText(frame, info, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 이미지 발행
            out_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.pub.publish(out_msg)
            
        except Exception as e:
            self.get_logger().error(f"추론 에러: {e}")

def main():
    rclpy.init()
    
    print("=" * 50)
    print(f"  🚀 YOLO TFLite 모드 (탐사 연동 가능)")
    print("=" * 50)
    print(f"  📁 모델: {MODEL_PATH}")
    print(f"  📡 JSON 토픽: /yolo_detections")
    print("=" * 50)
    
    try:
        node = YoloTfliteNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
