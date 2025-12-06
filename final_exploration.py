#!/usr/bin/env python3
"""
최종 탐사 노드 (Final Exploration)
- simple_exploration.py의 강력한 Frontier 탐사 로직 기반
- YOLO 박스 감지 시 즉시 탐사 중단 및 접근
- 박스 감지 시 추적하며 직진 접근

실행:
  1. YOLO 실행: python3 yolo_test1.py
  2. 이 노드 실행: python3 final_exploration.py
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from std_msgs.msg import String, Float32MultiArray
from visualization_msgs.msg import Marker
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from tf2_ros import Buffer, TransformListener
import numpy as np
import cv2
import time
import math
import json
from stop_utils import CmdStopper
from frontier_utils import compute_frontier_goal

# ================= [설정] =================
BOX_CLASS_NAME = "box"       # YOLO 클래스 이름 (모델에 맞게 수정)
BOX_BACK_OFFSET = 0.6        # 박스 뒤쪽으로 이동할 거리 (m)
CHECKED_BOX_RADIUS = 1.0     # 이미 검사한 박스 반경 (m)
YOLO_CONF_THRESHOLD = 0.75   # YOLO 신뢰도 임계값 (75%)
TARGET_BOX_COUNT = 2         # 목표 박스 개수
# ==========================================


class FinalExplorer(Node):
    def __init__(self):
        super().__init__('final_explorer')
        
        # 1. 구독자 설정
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.create_subscription(String, '/yolo_detections', self.yolo_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Float32MultiArray, '/fusion_box_point', self.fusion_callback, 10)
        
        # 2. 퍼블리셔
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(Marker, '/box_marker', 10)
        self.stopper = CmdStopper(self.cmd_vel_pub, spin_node=self)
        
        # 3. 데이터 변수
        self.map_data = None
        self.map_info = None
        self.front_distance = float('inf')
        self.last_scan = None
        
        # 4. 상태 변수
        self.is_navigating = False
        self.last_goal = None
        self.mode = "EXPLORE"  # "EXPLORE" or "APPROACH"
        self.aligning_to_box = False
        self.cancel_nav_requested = False
        
        # 5. 정렬 관련 (타임아웃/미감지 처리)
        self.align_start_time = None
        self.align_lost_count = 0
        
        # 6. 박스 관련
        self.box_detected = False
        self.checked_boxes = []  # [(x, y), ...] - 이미 간 박스 위치
        self.current_box_pos = None
        self.shutdown_requested = False
        
        # 7. 퓨전 박스 위치 (월드 좌표)
        self.fusion_box_world = None  # (x, y) - 월드 좌표 (map 프레임)
        self.fusion_box_timestamp = None  # 수신 시간
        
        # 8. TF (위치 추적용)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.get_logger().info("최종 탐사 노드 시작 (퓨전 모드)")

    # ===== 콜백 함수 =====
    def map_callback(self, msg):
        self.map_info = msg.info
        # occupancy grid를 numpy 배열로 변환
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))

    def scan_callback(self, msg):
        """전방 거리 측정 (박스 위치 추정용)
        마지막 스캔 보관하고 정면 +- 10 범위 최소거리를 front_distance로 갱신"""
        self.last_scan = msg
        if not msg.ranges:
            return
        # 정면 ±10도 거리 중 최소값
        mid = len(msg.ranges) // 2
        range_width = int(len(msg.ranges) * (20 / 360))  # 20도
        dists = msg.ranges[mid-range_width:mid+range_width]
        valid_dists = [d for d in dists if msg.range_min < d < msg.range_max]
        if valid_dists:
            self.front_distance = min(valid_dists)
        else:
            self.front_distance = 2.0  # 기본값

    def fusion_callback(self, msg):
        """fusion.py에서 계산한 박스 위치 수신 (이미 월드 좌표!)"""
        if len(msg.data) >= 2:
            world_x, world_y = msg.data[0], msg.data[1]
            self.fusion_box_world = (world_x, world_y)  # 월드 좌표로 저장
            self.fusion_box_timestamp = time.time()
            
            self.get_logger().info(
                f"🎯 퓨전 박스 수신 (월드 좌표): ({world_x:.2f}, {world_y:.2f})"
            )

    def local_to_world(self, local_x, local_y):
        """로봇 기준 좌표 → 월드 좌표 변환"""
        pose = self.get_robot_pose()
        if not pose:
            return None
        rx, ry, ryaw = pose
        
        # 회전 변환
        world_x = rx + local_x * math.cos(ryaw) - local_y * math.sin(ryaw)
        world_y = ry + local_x * math.sin(ryaw) + local_y * math.cos(ryaw)
        return world_x, world_y

    def yolo_callback(self, msg):
        """박스 감지 시 위치 저장 후 Nav2로 접근"""
        if self.shutdown_requested:
            return
        try:
            detections = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("JSON 파싱 실패")
            return
        
        self.get_logger().info(f"📩 YOLO 토픽 수신: {len(detections)}개 객체")
        
        # 가장 신뢰도 높은 박스 찾기
        best_box = None
        best_conf = YOLO_CONF_THRESHOLD
        for det in detections:
            name = det.get('name', '').lower()
            conf = det.get('conf', 0.0)
            if name == BOX_CLASS_NAME.lower() and conf >= best_conf:
                best_conf = conf
                best_box = det
        
        if not best_box:
            self.get_logger().warn(f"박스 클래스 없음 (threshold={YOLO_CONF_THRESHOLD})")
            return
        
        self.get_logger().info(f"✅ 박스 감지됨: conf={best_box.get('conf'):.2f}, angle={best_box.get('angle_deg'):.1f}deg")
        
        # 이미 접근 중이면 무시
        if self.mode == "APPROACH":
            self.get_logger().info("이미 APPROACH 모드 - 무시")
            return
        
        # === 박스 위치 추정 (퓨전 우선, 없으면 기존 방식) ===
        box_pos = None
        
        # 1. 퓨전 데이터가 최근 것이면 사용 (1초 이내)
        if self.fusion_box_world and self.fusion_box_timestamp:
            age = time.time() - self.fusion_box_timestamp
            if age < 1.0:
                box_pos = self.fusion_box_world  # 이미 월드 좌표!
                self.get_logger().info(f"📍 퓨전 기반 박스 위치 사용 (age={age:.2f}s)")
        
        # 2. 퓨전 데이터 없으면 기존 방식 (각도 + 라이다)
        if not box_pos:
            self.get_logger().info("📍 기존 방식 (YOLO 각도 + 라이다) 사용")
            box_pos = self.estimate_box_position(best_box)
        
        if not box_pos:
            self.get_logger().error("❌ 박스 위치 추정 실패!")
            return
        
        if self.is_checked_box(*box_pos):
            self.get_logger().info("이미 확인한 박스 - 무시")
            return  # 이미 간 박스는 무시
        
        # === 박스 발견 -> 멈추고 위치 저장 -> Nav2 APPROACH 모드! ===
        self.get_logger().info(f"박스 발견! 위치=({box_pos[0]:.2f}, {box_pos[1]:.2f}), 거리={self.front_distance:.2f}m")
        
        # 1. Nav2 취소
        self.cancel_nav()
        
        # 2. 정지 명령 (여러 번)
        self.stop_robot()
        
        # 3. 잠시 대기 (정지 확인)
        time.sleep(0.3)
        self.stop_robot()  # 한 번 더
        
        self.box_detected = True
        self.current_box_pos = box_pos
        self.mode = "APPROACH"

    # ===== 유틸리티 =====
    def get_robot_pose(self):
        try:
            # TF 도착 대기 (최대 0.5초)
            if not self.tf_buffer.can_transform('map', 'base_link', rclpy.time.Time(), timeout=Duration(seconds=0.5)):
                return None
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            q = t.transform.rotation
            yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))
            return t.transform.translation.x, t.transform.translation.y, yaw
        except Exception:
            return None

    def get_distance_along_angle(self, angle_rad):
        """
        개선된 거리 측정:
        단일 각도가 아니라, 해당 각도 주변(Cone)을 스캔하여
        가장 가까운 물체(박스일 확률 높음)의 거리를 반환
        """
        if self.last_scan is None:
            self.get_logger().warn("LiDAR 데이터 없음 - 거리 추정 실패")
            return None
            
        scan = self.last_scan
        angle_min = scan.angle_min
        angle_inc = scan.angle_increment
        
        # 1. YOLO 각도에 해당하는 라이다 인덱스 계산
        center_idx = int(round((angle_rad - angle_min) / angle_inc))
        
        # 2. 탐색 범위 설정 (예: ±10도) -> 라이다 인덱스 범위
        search_angle_deg = 10.0 
        search_width = int(math.radians(search_angle_deg) / angle_inc)
        
        start_idx = max(0, center_idx - search_width)
        end_idx = min(len(scan.ranges), center_idx + search_width + 1)
        
        # 3. 유효한 거리 데이터 추출
        valid_dists = []
        for r in scan.ranges[start_idx:end_idx]:
            if scan.range_min < r < scan.range_max:
                valid_dists.append(r)
                
        if not valid_dists:
            self.get_logger().warn("해당 각도 범위에 유효한 라이다 데이터 없음")
            return None
            
        # 4. 가장 가까운 거리 반환 (박스는 벽보다 앞에 튀어나와 있음)
        # 노이즈 방지를 위해 너무 가까운 값(0.1m 이하)은 제외할 수도 있음
        min_dist = min(valid_dists)
        
        # 디버깅용 로그
        self.get_logger().info(f"YOLO각도: {math.degrees(angle_rad):.1f} | 측정거리: {min_dist:.2f}m")
        
        return min_dist

    def estimate_box_position(self, detection):
        """YOLO가 준 각도 정보를 이용해 박스 월드 좌표 추정 (위치 추정 로직 보완)"""
        pose = self.get_robot_pose()
        if not pose:
            self.get_logger().warn("TF 조회 실패 - 로봇 위치 모름")
            return None

        angle_rad = detection.get('angle_rad')
        if angle_rad is None:
            self.get_logger().warn("YOLO 감지 데이터에 angle_rad 없음")
            return None

        # 거리 측정 (개선된 함수 사용)
        distance = self.get_distance_along_angle(angle_rad)
        
        # 거리가 너무 멀면(예: 3.5m 이상) 박스가 아니라 벽일 수 있음 -> 무시하거나 접근 보류
        if distance is None or distance > 3.5:
            self.get_logger().warn(f"측정된 거리가 너무 멀음 ({distance}m). 박스가 아닐 수 있음.")
            return None

        rx, ry, ryaw = pose
        
        # 월드 좌표 계산
        heading = ryaw + angle_rad
        bx = rx + distance * math.cos(heading)
        by = ry + distance * math.sin(heading)
        
        self.front_distance = distance
        self.get_logger().info(
            f"박스 추정: bearing={math.degrees(angle_rad):.1f}deg, 거리={distance:.2f}m, 위치=({bx:.2f}, {by:.2f})"
        )
        
        # Rviz 시각화 마커 발행
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = bx
        marker.pose.position.y = by
        marker.pose.position.z = 0.2
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0  # 빨간색 구체
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.marker_pub.publish(marker)
        
        return bx, by

    def is_checked_box(self, bx, by):
        """이미 확인한 박스인지 검사"""
        for cx, cy in self.checked_boxes:
            if math.hypot(bx-cx, by-cy) < CHECKED_BOX_RADIUS:
                return True
        return False
    
    def cancel_nav(self):
        """Nav2 제어 중단 요청"""
        self.is_navigating = False
        self.cancel_nav_requested = True
        self.get_logger().info("Nav2 취소 요청")
    
    def publish_cmd_vel(self, linear_x, angular_z):
        """cmd_vel 퍼블리시"""
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(msg)
    
    def stop_robot(self, duration_sec=0.5):
        """정지 헬퍼 호출"""
        self.get_logger().info("정지 명령 발행!")
        self.stopper.stop_now(duration_sec)

    def wait_with_spin(self, duration_sec):
        """spin을 유지하면서 대기 (콜백 처리 계속)"""
        end_time = time.time() + duration_sec
        while time.time() < end_time and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.05)  # CPU 과부하 방지

    def rotate_scan(self, duration_sec=8.0, angular_speed=0.5):
        """
        제자리에서 회전하며 YOLO로 박스 스캔
        - duration_sec: 회전 시간 (8초 ≈ 360도 at 0.5 rad/s)
        - angular_speed: 회전 속도 (rad/s)
        - 박스 발견 시 즉시 중단하고 True 반환
        """
        self.get_logger().info(f"🔄 회전 스캔 시작 ({duration_sec}초)")
        start_time = time.time()
        
        while time.time() - start_time < duration_sec and rclpy.ok():
            # 회전 명령
            self.publish_cmd_vel(0.0, angular_speed)
            
            # 콜백 처리 (YOLO 감지 확인)
            rclpy.spin_once(self, timeout_sec=0.05)
            
            # 박스 발견하면 중단
            if self.mode == "APPROACH":
                self.get_logger().info("🎯 회전 스캔 중 박스 발견! 스캔 중단")
                self.stop_robot()
                return True
            
            time.sleep(0.05)
        
        # 회전 완료 후 정지
        self.stop_robot()
        self.get_logger().info("🔄 회전 스캔 완료 (박스 미발견)")
        return False

    # ===== Frontier 로직 =====
def main():
    rclpy.init()
    node = FinalExplorer()
    nav = BasicNavigator()
    
    # Nav2 준비
    print("Nav2 준비 중...")
    nav.waitUntilNav2Active(localizer='slam_toolbox')
    print("준비 완료! 탐사 시작!")
    
    # 맵 대기
    while node.map_data is None:
        rclpy.spin_once(node, timeout_sec=1.0)
        print("맵 기다리는 중...")
    
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            
            # 강제 정지 유지 구간: 다른 cmd_vel을 덮어쓰기
            if node.stopper.enforce_stop():
                continue
            
            # Nav2 취소 요청 처리
            if node.cancel_nav_requested:
                nav.cancelTask()
                node.cancel_nav_requested = False
                # Nav2가 완전히 취소될 때까지 대기 (중요!)
                time.sleep(0.3)
                # 추가 정지 명령
                node.stop_robot()
                continue  # 이번 루프는 스킵하고 다음으로
            
            # === Nav2 접근 모드 (APPROACH) ===
            if node.mode == "APPROACH":
                if node.current_box_pos is None:
                    print("박스 위치 없음 -> 탐사 복귀")
                    node.mode = "EXPLORE"
                    node.box_detected = False
                    continue
                
                if not node.is_navigating:
                    bx, by = node.current_box_pos
                    
                    # 박스 앞 목표 지점 계산
                    pose = node.get_robot_pose()
                    if pose:
                        rx, ry, _ = pose
                        angle = math.atan2(by - ry, bx - rx)

                        # 박스 "뒤쪽"으로 오프셋 (로봇->박스 방향을 기준으로 박스 반대편)
                        tx = bx + BOX_BACK_OFFSET * math.cos(angle)
                        ty = by + BOX_BACK_OFFSET * math.sin(angle)
                        
                        # 박스를 향해 뒤에서 바라보도록 180도 회전
                        face_box = angle + math.pi
                        qz = math.sin(face_box / 2)
                        qw = math.cos(face_box / 2)
                        
                        print(f"[APPROACH] 박스 접근 시작!")
                        print(f"  현재 위치: ({rx:.2f}, {ry:.2f})")
                        print(f"  박스 위치: ({bx:.2f}, {by:.2f})")
                        print(f"  목표 위치: ({tx:.2f}, {ty:.2f})")
                        
                        # Nav2 goal 설정 전 잠시 대기
                        time.sleep(0.2)
                        
                        goal = PoseStamped()
                        goal.header.frame_id = 'map'
                        goal.header.stamp = nav.get_clock().now().to_msg()
                        goal.pose.position.x = tx
                        goal.pose.position.y = ty
                        goal.pose.orientation.z = qz
                        goal.pose.orientation.w = qw
                        
                        # Nav2 goal 전송
                        nav.goToPose(goal)
                        print(f"[APPROACH] Nav2 goal 전송 완료!")
                        
                        node.is_navigating = True
                        node.box_detected = False
                    else:
                        print("로봇 위치 불명 -> 탐사 복귀")
                        node.mode = "EXPLORE"
                        node.box_detected = False
                
                elif nav.isTaskComplete():
                    result = nav.getResult()
                    if result == TaskResult.SUCCEEDED:
                        is_final_box = (len(node.checked_boxes) + 1) >= TARGET_BOX_COUNT
                        if is_final_box:
                            print("박스 도착 완료! 목표 수량 달성.")
                        else:
                            print("박스 도착 완료! (3초 대기)")
                            node.wait_with_spin(3.0)  # spin 유지하면서 대기
                        
                        # 완료 처리
                        node.checked_boxes.append(node.current_box_pos)
                        print(f"박스 기록 완료 (총 {len(node.checked_boxes)}개)")
                        if len(node.checked_boxes) >= TARGET_BOX_COUNT:
                            print("박스 두 개 확인! 탐사를 종료합니다.")
                            node.shutdown_requested = True
                            node.stop_robot()
                    else:
                        print(f"박스 접근 실패: {result}")
                    
                    if node.shutdown_requested:
                        node.is_navigating = False
                        node.current_box_pos = None
                        node.box_detected = False
                        break
                    
                    print("탐사 모드 복귀")
                    node.mode = "EXPLORE"
                    node.is_navigating = False
                    node.current_box_pos = None
                    node.box_detected = False

            # === 탐사 모드 (EXPLORE) ===
            elif node.mode == "EXPLORE":
                if not node.is_navigating:
                    target = compute_frontier_goal(node.map_data, node.map_info, node.last_goal)
                    if target:
                        tx, ty = target
                        print(f"\n탐사 목표: ({tx:.2f}, {ty:.2f})")
                        
                        goal = PoseStamped()
                        goal.header.frame_id = 'map'
                        goal.header.stamp = nav.get_clock().now().to_msg()
                        goal.pose.position.x = tx
                        goal.pose.position.y = ty
                        goal.pose.orientation.w = 1.0
                        
                        nav.goToPose(goal)
                        node.is_navigating = True
                        node.last_goal = (tx, ty)
                    else:
                        print("더 이상 갈 곳이 없음 (탐사 완료)")
                        node.wait_with_spin(2.0)  # spin 유지하면서 대기
                
                elif nav.isTaskComplete():
                    # 성공이든 실패든 다음 목표 찾기
                    result = nav.getResult()
                    if result != TaskResult.SUCCEEDED:
                        print(f"탐사 목표 도달 실패: {result}")
                    node.is_navigating = False
                    
                    # 🔄 Frontier 도착 후 회전 스캔 (박스 찾기)
                    if node.mode == "EXPLORE":  # APPROACH로 전환 안 됐으면
                        print("🔄 주변 박스 스캔 시작...")
                        found = node.rotate_scan(duration_sec=6.0, angular_speed=0.6)
                        if found:
                            print("박스 발견! APPROACH 모드로 전환됨")
                            continue

    except KeyboardInterrupt:
        print("\n사용자 종료")

    if node.shutdown_requested:
        print("목표 박스 두 개 확보 완료. 노드를 종료합니다.")

    nav.lifecycleShutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
