#!/usr/bin/env python3

import cv2
import rclpy
import numpy as np
import json
import math
import time
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import LaserScan, CameraInfo, Image
from std_msgs.msg import String # JSON 수신용
from sklearn.cluster import DBSCAN
from cv_bridge import CvBridge
from std_msgs.msg import Float32MultiArray


class LidarCameraProjector(Node):
    def __init__(self):
        super().__init__("lidar_camera_projector")

        self.get_logger().info("node start")

        self.bridge = CvBridge()

        # TF 버퍼 & 리스너
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.EPSILON = 0.1   # 클러스터를 형성할 최대 거리 (m)
        self.MIN_POINTS = 15

        # 발행자
        self.fusion_pub = self.create_publisher(Image, "/fusion", 10)
        self.fusion_box_pt_pub = self.create_publisher(Float32MultiArray, "/fusion_box_point", 10)
        self.filtered_scan_pub = self.create_publisher(LaserScan, "/filtered_scan", 10) # <-- 추가: 필터링된 스캔 발행

        # 토픽 구독/발행
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.cb_scan, 10)
        self.sub_cam  = self.create_subscription(CameraInfo, "/oakd/rgb/preview/camera_info", self.cb_camera, 10)
        self.sub_img  = self.create_subscription(Image, "/yolo_result", self.cb_image, 10) # yolo_test1.py의 결과 이미지 토픽으로 변경
        self.sub_yolo = self.create_subscription(String, "/yolo_detections", self.cb_yolo_detections, 10) # <-- 추가: YOLO JSON 수신

        # Intrinsic / Distortion / Image 저장 버퍼
        self.K = None        # 3×3 intrinsic
        self.D = None        # distortion coefficients
        self.latest_image = None
        self.latest_scan = None
        self.frame_camera = "oakd_rgb_camera_optical_frame"  # ← lidar를 이 프레임으로 변환
        self.frame_lidar  = "rplidar_link"           # scan에서 읽어올 frame_id

        # YOLO 감지 정보 버퍼
        self.latest_bboxes = [] # [{'box': [x1, y1, x2, y2], 'name': 'box', ...}, ...]
        self.last_bbox_time = 0.0  # 최근 bbox 수신 시각

    def cb_yolo_detections(self, msg: String):
        """YOLO 감지 결과를 JSON으로 수신"""
        try:
            detections = json.loads(msg.data)
            # YOLO 노드에서 발행하는 박스 좌표는 평균화된 좌표이므로
            # 이 노드에서는 raw det_pub을 구독하는 것이 더 정확할 수 있으나,
            # 여기서는 yolo_test1.py의 /yolo_detections 토픽 (평균화)을 구독한다고 가정합니다.
            self.latest_bboxes = [det for det in detections if det.get('name', '').lower() == 'box'] # 박스만 필터링
            if self.latest_bboxes:
                self.last_bbox_time = time.time()
            else:
                self.last_bbox_time = 0.0
            print(self.latest_bboxes)
        except json.JSONDecodeError:
            self.get_logger().warn("JSON 파싱 실패")
            self.latest_bboxes = []
            self.last_bbox_time = 0.0
    
    def tf_to_matrix(self, tf: TransformStamped):
        """TransformStamped → 4×4 변환 행렬"""
        t = tf.transform.translation
        q = tf.transform.rotation

        x, y, z, w = q.x, q.y, q.z, q.w

        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),         2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),         1 - 2 * (x * x + y * y)]
        ])

        T = np.eye(4)
        T[:3, :3] = R
        T[0, 3] = t.x
        T[1, 3] = t.y
        T[2, 3] = t.z
        return T

    def cb_camera(self, msg: CameraInfo):
        """Intrinsic + Distortion 저장"""
        self.K = np.array(msg.k).reshape(3,3)
        self.D = np.array(msg.d)
        
    def cb_image(self, msg: Image):
        """이미지 저장"""
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # self.latest_image = cv_img.copy()

        """scan 점을 camera image에 투영"""
        if self.K is None or self.latest_scan is None:
            return  # 아직 카메라정보 or 이미지 없음

        self.frame_lidar = self.latest_scan.header.frame_id

        # 1) TF lookup (scan → camera)
        try:
            tf = self.tf_buffer.lookup_transform(self.frame_camera, self.frame_lidar, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        T_lidar_cam = self.tf_to_matrix(tf)  # 4×4

        # 2) 2D LiDAR scan → 3D points (라이다 좌표계) & 인덱스 저장
        pts_lidar = [] # [x, y, z, r, index]
        angle = self.latest_scan.angle_min
        for i, r in enumerate(self.latest_scan.ranges):
            if np.isfinite(r):
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                z = 0.0
                pts_lidar.append([x,y,z,r,i])  # r(dist), i(original index)도 저장
            angle += self.latest_scan.angle_increment
        
        pts_lidar = np.array(pts_lidar)

        if len(pts_lidar) == 0:
            return

        # 3) LiDAR 점 → Camera 좌표계로 변환
        xyz = pts_lidar[:, :3]
        dist = pts_lidar[:, 3]
        indices = pts_lidar[:, 4].astype(int)

        xyz_h = np.hstack((xyz, np.ones((len(xyz),1))))
        xyz_cam = (T_lidar_cam @ xyz_h.T).T[:, :3]
        
        # 4) Intrinsic로 2D 투영
        uv_h = (self.K @ xyz_cam.T).T
        
        # z>0 (카메라 앞에 있는 점)만 유효
        valid_mask = uv_h[:, 2] > 0
        uv = (uv_h[valid_mask, :2] / uv_h[valid_mask, 2:3])
        
        # 유효한 점들에 대한 정보 필터링
        dist_valid = dist[valid_mask]
        indices_valid = indices[valid_mask]
        
        # 이미지 크기
        img_h, img_w = cv_img.shape[:2]

        # 5) 이미지 overlay (거리 기반 색) & BBox 필터링
        overlay = cv_img.copy()
        
        # 필터링된 스캔 데이터를 위한 버퍼 (기존 스캔 크기)
        filtered_ranges = [float('inf')] * len(self.latest_scan.ranges)
        bbox_hits = 0

        # BBox 정보 (x1, y1, x2, y2)
        # 오래된 감지는 무시 (예: 0.6초 초과)
        if (not self.latest_bboxes) or (time.time() - self.last_bbox_time > 0.6):
            self.latest_bboxes = []
            self.last_bbox_time = 0.0
            # BBox 정보가 없으면, 퓨전 이미지만 발행하고 필터링 스캔은 건너뜀
            pass 
        else:
            # 신뢰도(conf) 가장 높은 박스 선택
            best_bbox = max(self.latest_bboxes, key=lambda b: b.get("conf", 0.0))
            x1, y1, x2, y2 = best_bbox.get('box') # YOLO 노드가 이 키를 보장해야 함

            for i, (u, v) in enumerate(uv):
                # 퓨전 이미지에 투영
                if 0 <= u < img_w and 0 <= v < img_h:
                    
                    # scale 0~255 (멀수록 255=white)
                    # 10m 이상은 완전 흰색에 가깝게 스케일링
                    max_dist = 5.0 # 시각화 거리 기준 5m로 변경
                    intensity = np.clip(dist_valid[i] / max_dist * 255.0, 0, 255)
                    c = int(intensity)
                    
                    # 퓨전 이미지에 점 그리기
                    cv2.circle(overlay, (int(u), int(v)), 2, (c,c,c), -1)

                    # BBox 필터링 로직
                    # BBox 내부에 점이 위치하는지 확인
                    if x1 <= u <= x2 and y1 <= v <= y2:
                        
                        # 6) BBox 내부에 들어온 점은 필터링된 스캔에 기록
                        original_index = indices_valid[i]
                        # 거리가 유효하면 최소값으로 업데이트 (하나의 BBox에 여러 점이 들어올 수 있음)
                        if filtered_ranges[original_index] > dist_valid[i]:
                             filtered_ranges[original_index] = dist_valid[i]
                        
                        # 시각화: 필터링된 점은 다른 색으로 표시 (예: 노란색)
                        cv2.circle(overlay, (int(u), int(v)), 3, (0, 255, 255), -1)
                        bbox_hits += 1

        # 7) 필터링된 스캔 발행
        if self.latest_bboxes:
            self.get_logger().info(f"BBox 내 LiDAR 점 감지: {bbox_hits}개")
            
            if bbox_hits == 0:
                self.get_logger().info("BBox 있지만 LiDAR 히트 0 → 발행/클러스터 건너뜀")
                # 더 이상 재사용되지 않도록 BBox 비우기
                self.latest_bboxes = []
                self.last_bbox_time = 0.0
                return

            filtered_msg = LaserScan()
            filtered_msg.header = self.latest_scan.header
            filtered_msg.angle_min = self.latest_scan.angle_min
            filtered_msg.angle_max = self.latest_scan.angle_max
            filtered_msg.angle_increment = self.latest_scan.angle_increment
            filtered_msg.time_increment = self.latest_scan.time_increment
            filtered_msg.scan_time = self.latest_scan.scan_time
            filtered_msg.range_min = self.latest_scan.range_min
            filtered_msg.range_max = self.latest_scan.range_max
            
            # inf는 그대로 유지하여 "측정 없음"으로 전달
            filtered_msg.ranges = [
                r if r != float('inf') else float('inf')
                for r in filtered_ranges
            ]

            # YOLO 노드의 angle_deg/rad 정보를 ranges에 반영할 방법이 없으므로,
            # intensity 필드에 박스의 각도 정보를 넣거나, 
            # 아니면 /filtered_scan과 /yolo_detections 토픽을 같이 사용하는 것이 더 일반적입니다.
            # 여기서는 유효한 점의 ranges만 넣고 발행합니다.
            self.clustering(filtered_msg)

            self.filtered_scan_pub.publish(filtered_msg)
            
        # 8) 퓨전 이미지 발행
        imgmsg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        self.fusion_pub.publish(imgmsg)

    def lidar_to_map(self, local_x, local_y):
        """라이다 좌표계 → map 좌표계 변환"""
        try:
            # 라이다 프레임 → map 프레임 TF 조회
            tf = self.tf_buffer.lookup_transform('map', self.frame_lidar, rclpy.time.Time())
            
            # 변환 행렬 생성
            T = self.tf_to_matrix(tf)
            
            # 라이다 좌표를 동차 좌표로 변환
            pt_lidar = np.array([local_x, local_y, 0.0, 1.0])
            
            # map 좌표로 변환
            pt_map = T @ pt_lidar
            
            return float(pt_map[0]), float(pt_map[1])
        except Exception as e:
            self.get_logger().warn(f"라이다→map 변환 실패: {e}")
            return None

    def clustering(self, filtered_scan: LaserScan):
        X = []
        angle = filtered_scan.angle_min
        for i, r in enumerate(filtered_scan.ranges):
            # 유효한 거리만 사용 (0보다 크고 유한한 값)
            if r > 0.05 and np.isfinite(r):
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                X.append([x, y])
            angle += filtered_scan.angle_increment
        
        # 최소 점 개수 체크
        if len(X) < self.MIN_POINTS:
            self.get_logger().info(f"유효 점 부족: {len(X)}개 (최소 {self.MIN_POINTS}개 필요)")
            return
        
        # 리스트를 numpy 배열로 변환
        X = np.array(X)
        
        db = DBSCAN(eps=self.EPSILON, min_samples=self.MIN_POINTS).fit(X)
        labels = db.labels_

        if 0 in labels:
            cluster_0_mask = (labels == 0)
            cluster_0_points = X[cluster_0_mask]
            
            # NumPy의 np.mean 함수를 사용하여 평균을 한 번에 계산 (라이다 좌표계)
            local_x, local_y = np.mean(cluster_0_points, axis=0)
            
            # 라이다 좌표 → map 좌표 변환
            map_pos = self.lidar_to_map(local_x, local_y)
            
            if map_pos:
                map_x, map_y = map_pos
                self.get_logger().warn(
                    f"🎯 박스 위치: 라이다=({local_x:.2f}, {local_y:.2f}), "
                    f"월드=({map_x:.2f}, {map_y:.2f}), 점={len(cluster_0_points)}개"
                )
                
                # 월드 좌표로 발행!
                fusion_box_pt = Float32MultiArray()
                fusion_box_pt.data = [map_x, map_y]
                self.fusion_box_pt_pub.publish(fusion_box_pt)
            else:
                self.get_logger().warn(f"박스 감지했으나 좌표 변환 실패")

        else:
            self.get_logger().info("박스 클러스터 없음")

        self.get_logger().info(f"클러스터 라벨: {np.unique(labels)}")

    def cb_scan(self, msg: LaserScan):
        self.latest_scan = msg

def main():
    rclpy.init()
    node = LidarCameraProjector()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()