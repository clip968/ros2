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
from sensor_msgs.msg import LaserScan, CameraInfo, Image, CompressedImage
from std_msgs.msg import String # JSON 수신용
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
import message_filters

def quaternion_to_matrix(q):
    """쿼터니언 [x, y, z, w] → 4x4 변환 행렬 (tf_transformations 대체)"""
    x, y, z, w = q
    
    # 회전 행렬 계산
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])
    
    # 4x4 행렬로 확장
    T = np.eye(4)
    T[:3, :3] = R
    return T


class LidarCameraProjector(Node):
    def __init__(self):
        super().__init__("lidar_camera_projector")

        self.get_logger().info("node start")

        self.bridge = CvBridge()

        # TF 버퍼 & 리스너
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.EPSILON = 0.5   # 클러스터를 형성할 최대 거리 (m)
        self.MIN_POINTS = 5

        # 발행자
        self.fusion_pub = self.create_publisher(Image, "/fusion", 10)
        self.fusion_box_pts_pub = self.create_publisher(PoseArray, "/fusion_box_point", 10)
        self.filtered_scan_pub = self.create_publisher(LaserScan, "/filtered_scan", 10)
        self.bbox_map_pub = self.create_publisher(PoseStamped, "/bbox_map", 10)

        # 토픽 구독 (message_filters)
        self.sub_scan = message_filters.Subscriber(self, LaserScan, "/scan")
        self.sub_yolo = message_filters.Subscriber(self, String, "/yolo_detections")
        self.sub_cam  = message_filters.Subscriber(self, CameraInfo, "/oakd/rgb/preview/camera_info")

        # 동기화 (ApproximateTime)
        # 큐 크기 10, slop 0.1초 (필요시 조정)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_scan, self.sub_yolo, self.sub_cam], 
            queue_size=10, slop=0.05
        )
        self.ts.registerCallback(self.cb_sync)

        # 이미지 구독 (시각화용, 비동기)
        self.sub_img  = self.create_subscription(CompressedImage, "/yolo_result", self.cb_image, 10)

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

    def cb_sync(self, scan_msg, yolo_msg, cam_info_msg):
        """동기화된 콜백: Scan + YOLO + CameraInfo"""
        
        # 1. 데이터 파싱 및 저장
        self.latest_scan = scan_msg
        self.K = np.array(cam_info_msg.k).reshape(3,3)
        self.D = np.array(cam_info_msg.d)
        
        try:
            detections = json.loads(yolo_msg.data)
            self.latest_bboxes = [det for det in detections if det.get('name', '').lower() == 'box']
            if self.latest_bboxes:
                self.last_bbox_time = time.time()
            else:
                self.last_bbox_time = 0.0
        except json.JSONDecodeError:
            self.get_logger().warn("JSON 파싱 실패")
            self.latest_bboxes = []
            self.last_bbox_time = 0.0

        # 2. Main Processing (Projection & Filtering)
        # TF lookup
        self.frame_lidar = scan_msg.header.frame_id
        try:
            # 시간 동기화된 메시지들이므로 해당 stamp 사용 시도 가능하나,
            # TF 트리는 약간의 지연이 있을 수 있으므로 최신(Time())을 조회하거나 scan_msg.header.stamp를 사용
            # 여기서는 편의상 최신 TF 조회
            tf = self.tf_buffer.lookup_transform(self.frame_camera, self.frame_lidar, rclpy.time.Time())
        except Exception as e:
            # self.get_logger().warn(f"TF lookup failed: {e}")
            return

        T_lidar_cam = self.tf_to_matrix(tf)

        # 2D LiDAR -> 3D Points
        ranges = np.array(scan_msg.ranges, dtype=np.float32)
        num = len(ranges)
        angles = scan_msg.angle_min + np.arange(num) * scan_msg.angle_increment
        valid_mask = np.isfinite(ranges) & (ranges < 5.0)
        
        r_valid = ranges[valid_mask]
        a_valid = angles[valid_mask]
        idx_valid = np.nonzero(valid_mask)[0]

        x = r_valid * np.cos(a_valid)
        y = r_valid * np.sin(a_valid)
        z = np.zeros_like(x)
        
        pts_lidar = np.column_stack((x, y, z, r_valid, idx_valid))
        if len(pts_lidar) == 0:
            return

        # 3D Project to Camera Plane
        xyz = pts_lidar[:, :3]
        dist = pts_lidar[:, 3]
        indices = pts_lidar[:, 4].astype(int)

        xyz_h = np.hstack((xyz, np.ones((len(xyz), 1))))
        xyz_cam = (T_lidar_cam @ xyz_h.T).T[:, :3]
        
        uv_h = (self.K @ xyz_cam.T).T
        
        # z > 0 check
        valid_z = uv_h[:, 2] > 0
        uv = (uv_h[valid_z, :2] / uv_h[valid_z, 2:3])
        
        dist_valid_z = dist[valid_z]
        indices_valid_z = indices[valid_z]

        # 필터링 결과 버퍼
        filtered_ranges = [float('inf')] * num

        bbox_msg = PoseArray()
        bbox_msg.header.stamp = scan_msg.header.stamp
        bbox_msg.header.frame_id = "map"

        if self.latest_bboxes:
            for i, bbox in enumerate(self.latest_bboxes):
                bx1, by1, bx2, by2 = bbox.get('box')
                
                in_bbox = (bx1 <= uv[:, 0]) & (uv[:, 0] <= bx2) & (by1 <= uv[:, 1]) & (uv[:, 1] <= by2)
                if not in_bbox.any():
                    continue
                
                indicies = indices_valid_z[in_bbox]
                dists = dist_valid_z[in_bbox]

                filtered_ranges[indicies] = dists

                angles = scan_msg.angle_min + indicies * scan_msg.angle_increment
                x = dists * np.cos(angles)
                y = dists * np.sin(angles)

                cx, cy = self.lidar_to_map(
                    np.median(x), 
                    np.median(y), 
                    scan_msg.header.stamp)

                if cx is None or cy is None:
                    continue

                pose = Pose()
                pose.position.x = cx
                pose.position.y = cy
                pose.position.z = 0.0

                pose.orientation.x = 0.0
                pose.orientation.y = 0.0
                pose.orientation.z = 0.0
                pose.orientation.w = 1.0

                bbox_msg.poses.append(pose)

        # 3. Publish results
        if len(bbox_msg.poses) > 0:
            filtered_msg = LaserScan()
            filtered_msg.header = scan_msg.header
            filtered_msg.angle_min = scan_msg.angle_min
            filtered_msg.angle_max = scan_msg.angle_max
            filtered_msg.angle_increment = scan_msg.angle_increment
            filtered_msg.time_increment = scan_msg.time_increment
            filtered_msg.scan_time = scan_msg.scan_time
            filtered_msg.range_min = scan_msg.range_min
            filtered_msg.range_max = scan_msg.range_max
            filtered_msg.ranges = [
                r if r != float('inf') else float('inf')
                for r in filtered_ranges
            ]
            
            self.filtered_scan_pub.publish(filtered_msg)
            self.fusion_box_pts_pub.publish(bbox_msg)

    def lidar_to_map(self, local_x, local_y, time):
        """라이다 좌표계 → map 좌표계 변환"""
        try:
            # 라이다 프레임 → map 프레임 TF 조회
            tf = self.tf_buffer.lookup_transform('map', self.frame_lidar, time)
        

            # 변환 행렬 생성
            T = self.tf_to_matrix(tf)
            
            # 라이다 좌표를 동차 좌표로 변환
            pt_lidar = np.array([local_x, local_y, 0.0, 1.0])
            
            # map 좌표로 변환
            pt_map = T @ pt_lidar
            
            return float(pt_map[0]), float(pt_map[1])

        except Exception as e:
            self.get_logger().warn(f"라이다→map 변환 실패: {e}")
            return None, None

    def tf_to_matrix(self, tf: TransformStamped):
        """TransformStamped → 4×4 변환 행렬"""
        t = tf.transform.translation
        q = tf.transform.rotation

        # quaternion to 4×4 matrix (직접 계산)
        T = quaternion_to_matrix([q.x, q.y, q.z, q.w])

        # translation 추가
        T[0, 3] = t.x
        T[1, 3] = t.y
        T[2, 3] = t.z
        return T

    def cb_image(self, msg: CompressedImage):
        """이미지 저장 및 시각화 (Fusion Overlay)"""
        cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # 데이터가 없으면 그냥 원본 이미지 발행
        if self.K is None or self.latest_scan is None:
            imgmsg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
            self.fusion_pub.publish(imgmsg)
            return

        # ---- 시각화 로직 (cb_sync와 유사하지만 시각화 전용) ----
        # 이미지가 들어온 시점의 최신 데이터들을 이용해 그리기
        
        # 1) TF lookup
        try:
            tf = self.tf_buffer.lookup_transform(self.frame_camera, self.frame_lidar, rclpy.time.Time())
        except Exception:
            return

        T_lidar_cam = self.tf_to_matrix(tf)
        
        ranges = np.array(self.latest_scan.ranges, dtype=np.float32)
        angles = self.latest_scan.angle_min + np.arange(len(ranges)) * self.latest_scan.angle_increment
        valid_mask = np.isfinite(ranges) & (ranges < 50.0) # 시각화는 좀 멀리까지 보여도 됨
        
        r_valid = ranges[valid_mask]
        a_valid = angles[valid_mask]
        
        x = r_valid * np.cos(a_valid)
        y = r_valid * np.sin(a_valid)
        z = np.zeros_like(x)
        
        xyz_h = np.column_stack((x, y, z, np.ones_like(x)))
        xyz_cam = (T_lidar_cam @ xyz_h.T).T[:, :3]
        
        uv_h = (self.K @ xyz_cam.T).T
        valid_z = uv_h[:, 2] > 0
        uv = (uv_h[valid_z, :2] / uv_h[valid_z, 2:3])
        dist_valid = r_valid[valid_z] # 원점 거리 근사

        overlay = cv_img.copy()
        img_h, img_w = cv_img.shape[:2]

        # bbox 그리기 (옵션)
        # for bbox in self.latest_bboxes:
        #     bx1, by1, bx2, by2 = bbox.get('box')
        #     cv2.rectangle(overlay, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 0, 255), 2)

        # 점 그리기
        for i, (u, v) in enumerate(uv):
            if 0 <= u < img_w and 0 <= v < img_h:
                d = dist_valid[i]
                
                # 색상
                max_dist = 10.0
                intensity = np.clip(d / max_dist * 255.0, 0, 255)
                c = int(intensity)
                color = (c, c, c)

                # 박스 내부 확인
                in_box = False
                for bbox in self.latest_bboxes:
                    bx1, by1, bx2, by2 = bbox.get('box')
                    if bx1 <= u <= bx2 and by1 <= v <= by2:
                        in_box = True
                        break
                
                if in_box:
                    color = (0, 255, 255) # Yellow inside box
                    cv2.circle(overlay, (int(u), int(v)), 3, color, -1)
                else:
                    cv2.circle(overlay, (int(u), int(v)), 1, color, -1)
        
        imgmsg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        self.fusion_pub.publish(imgmsg)

def main():
    rclpy.init()
    node = LidarCameraProjector()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()