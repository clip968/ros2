#!/usr/bin/env python3
"""
최종 탐사 노드 (Final Exploration) - 멀티스레드 비동기 버전
- MultiThreadedExecutor로 비동기 데이터 수신
- ReentrantCallbackGroup으로 병렬 콜백 처리
- 별도 스레드에서 spin 실행하여 blocking 작업 중에도 데이터 계속 수신

실행:
  1. YOLO 실행: python3 yolo_test1.py
  2. 이 노드 실행: python3 final_exploration_async.py
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
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
import threading
from stop_utils import CmdStopper
from frontier_utils import compute_frontier_goal

# ================= [설정] =================
BOX_CLASS_NAME = "box"       # YOLO 클래스 이름 (모델에 맞게 수정)
BOX_DEPTH = 0.3              # 박스 깊이 추정 (m) - 박스를 통과하기 위한 값
BOX_BEHIND_OFFSET = 0.5      # 박스 뒤쪽에서 떨어질 거리 (m)
CHECKED_BOX_RADIUS = 1.0     # 이미 검사한 박스 반경 (m)
YOLO_CONF_THRESHOLD = 0.75   # YOLO 신뢰도 임계값 (75%)
TARGET_BOX_COUNT = 2         # 목표 박스 개수
BOX_MEASURE_COUNT = 5        # 박스 좌표 측정 횟수 (중간값용)
BOX_MEASURE_INTERVAL = 0.3   # 측정 간격 (초)
# ==========================================


class FinalExplorerAsync(Node):
    def __init__(self):
        super().__init__('final_explorer_async')
        
        # 멀티스레드를 위한 CallbackGroup 생성
        self.callback_group = ReentrantCallbackGroup()
        
        # Thread-safe를 위한 Lock
        self.data_lock = threading.Lock()
        
        # 1. 구독자 설정 (모두 callback_group에 배치)
        self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10,
            callback_group=self.callback_group
        )
        self.create_subscription(
            Float32MultiArray, '/fusion_box_point', self.fusion_callback, 10,
            callback_group=self.callback_group
        )
        self.create_subscription(
            String, '/yolo_detections', self.yolo_callback, 10,
            callback_group=self.callback_group
        )
        
        # 2. 퍼블리셔
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(Marker, '/box_marker', 10)
        self.target_point_pub = self.create_publisher(PoseStamped, '/target_point', 10)
        self.stopper = CmdStopper(self.cmd_vel_pub, spin_node=None)  # spin_node는 None (별도 스레드에서 spin)
        
        # 3. 데이터 변수
        self.map_data = None
        self.map_info = None
        self.last_scan = None
        
        # 4. 상태 변수
        self.is_navigating = False
        self.last_goal = None
        self.mode = "EXPLORE"  # "EXPLORE" or "APPROACH"
        
        # 5. 박스 관련
        self.box_detected = False
        self.checked_boxes = []  # [(x, y), ...] - 이미 간 박스 위치
        self.current_box_pos = None
        self.shutdown_requested = False
        
        # 6. 퓨전 박스 위치 (월드 좌표)
        self.fusion_box_world = None  # (x, y) - 월드 좌표 (map 프레임)
        self.fusion_box_timestamp = None  # 수신 시간
        
        # 7. 박스 측정 버퍼 (여러 번 측정용)
        self.box_measurements = []  # [(x, y), ...]
        self.measuring_box = False  # 측정 중 플래그
        
        # 8. YOLO 감지 디버깅
        self.yolo_detections = []
        self.yolo_timestamp = None
        
        # 9. TF (위치 추적용)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.get_logger().info("🚀 최종 탐사 노드 시작 (멀티스레드 비동기 모드)")

        self.nav = BasicNavigator()


    # ===== 콜백 함수 (Thread-safe) =====
    def map_callback(self, msg):
        with self.data_lock:
            self.map_info = msg.info
            # occupancy grid를 numpy 배열로 변환
            self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))

    def yolo_callback(self, msg):
        """YOLO 감지 디버깅용 - 박스가 실제로 감지되는지 확인"""
        try:
            detections = json.loads(msg.data)
            box_detections = [det for det in detections if det.get('name', '').lower() == 'box']
            if box_detections:
                with self.data_lock:
                    self.yolo_detections = box_detections
                    self.yolo_timestamp = time.time()
                self.get_logger().info(f"🔍 YOLO 박스 {len(box_detections)}개 감지! (최고신뢰도: {max([d.get('conf', 0) for d in box_detections]):.2f})")
        except:
            pass

    def fusion_callback(self, msg):
        with self.data_lock:
            if self.mode == "APPROACH":
                self.get_logger().info("이미 APPROACH 모드 - 무시")
                return

            self.get_logger().info('🎯 퓨전 데이터 수신!')
            world_x, world_y = msg.data[0], msg.data[1]
            self.fusion_box_world = (world_x, world_y)  # 월드 좌표로 저장
            self.fusion_box_timestamp = time.time()
            
            self.get_logger().info(
                f"🎯 퓨전 박스 수신 (월드 좌표): ({world_x:.2f}, {world_y:.2f})"
            )

    # ===== 유틸리티 =====
    def get_robot_pose(self):
        try:
            # TF 도착 대기 (최대 0.1초로 단축 - 네트워크 부하 감소)
            if not self.tf_buffer.can_transform('map', 'base_link', rclpy.time.Time(), timeout=Duration(seconds=0.1)):
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
        min_dist = min(valid_dists)
        
        # 디버깅용 로그
        self.get_logger().info(f"YOLO각도: {math.degrees(angle_rad):.1f} | 측정거리: {min_dist:.2f}m")
        
        return min_dist

    def publish_cmd_vel(self, linear_x, angular_z):
        """cmd_vel 퍼블리시"""
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(msg)
    
    def stop_robot(self, duration_sec=0.5):
        """로봇 정지 (Nav2 취소 + cmd_vel 0,0 발행)"""
        self.nav.cancelTask()
        time.sleep(0.1)
        self.stopper.stop_now(duration_sec)

    def wait_async(self, duration_sec):
        """
        비동기 대기 (멀티스레드 버전)
        spin_once 불필요! 백그라운드에서 자동으로 콜백 실행됨
        """
        time.sleep(duration_sec)

    def rotate_scan(self, duration_sec=10.0, angular_speed=0.3):
        """
        제자리에서 회전하며 YOLO로 박스 스캔
        - duration_sec: 회전 시간 (10초 ≈ 180도 at 0.3 rad/s)
        - angular_speed: 회전 속도 (rad/s) - 낮을수록 천천히 회전
        - 박스 발견 시 즉시 중단하고 True 반환
        
        멀티스레드 버전: spin_once 불필요, 백그라운드에서 자동 콜백 처리
        """
        self.get_logger().info(f"🔄 회전 스캔 시작 ({duration_sec}초)")
        
        for i in range(15):
            self.get_logger().info(f"========== step : {i} ==========")
            self.stop_robot(0.3)

            # 5초 동안 대기 (백그라운드에서 자동으로 콜백 실행됨!)
            self.get_logger().info("⏳ 5초 대기 중... (백그라운드 자동 수신)")
            self.wait_async(5.0)
            
            # Thread-safe 데이터 읽기
            with self.data_lock:
                yolo_ts = self.yolo_timestamp
                yolo_count = len(self.yolo_detections)
                fusion_ts = self.fusion_box_timestamp
            
            # YOLO 감지 상태 확인
            if yolo_ts is not None:
                yolo_age = time.time() - yolo_ts
                self.get_logger().info(f"📸 YOLO: {yolo_count}개 박스 (최근 감지: {yolo_age:.2f}초 전)")
            else:
                self.get_logger().info("📸 YOLO: 감지 없음")
            
            # fusion 데이터 상태 확인
            if fusion_ts is not None:
                fusion_age = time.time() - fusion_ts
                self.get_logger().info(f"🎯 FUSION: 데이터 있음 ({fusion_age:.2f}초 전)")
                print(f"Fusion timestamp 차이: {fusion_age}")
            else:
                self.get_logger().info("🎯 FUSION: 데이터 없음")

            # fusion 데이터가 최근(1초 이내)이면 박스 발견으로 판정
            if fusion_ts is not None and (time.time() - fusion_ts) < 1.0:
                self.get_logger().info("✅ 회전 스캔 중 박스 발견! 스캔 중단")                
                return True

            # 다음 스텝: 약간 회전
            target_angle = np.deg2rad(10)
            angular_speed = 0.3  # rad/s
            rotate_duration = abs(target_angle) / angular_speed
            end_rot_time = time.time() + rotate_duration
            
            self.get_logger().info(f"🔄 {np.rad2deg(target_angle):.1f}도 회전 시작...")
            while time.time() < end_rot_time and rclpy.ok():
                # 회전 명령 퍼블리시
                self.publish_cmd_vel(0.0, np.sign(target_angle) * angular_speed)
                time.sleep(0.05)
            
            # 회전 후 정지 펄스
            self.publish_cmd_vel(0.0, 0.0)
        
        # 회전 완료 후 정지
        self.stop_robot()
        self.get_logger().info("🔄 회전 스캔 완료 (박스 미발견)")
        return False


def main():
    rclpy.init()
    
    # 멀티스레드 Executor 생성
    executor = MultiThreadedExecutor(num_threads=4)
    node = FinalExplorerAsync()
    executor.add_node(node)
    
    # 별도 스레드에서 executor.spin 실행 (백그라운드 콜백 처리)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    # Nav2 준비
    print("Nav2 준비 중...")
    node.nav.waitUntilNav2Active(localizer='slam_toolbox')
    print("준비 완료! 탐사 시작!")
    
    # 맵 대기
    while node.map_data is None:
        time.sleep(1.0)
        print("맵 기다리는 중...")
    
    try:
        while rclpy.ok():
            time.sleep(0.1)
            
            # === Nav2 접근 모드 (APPROACH) ===
            if node.mode == "APPROACH":
                if node.current_box_pos is None:
                    print("박스 위치 없음 -> 탐사 복귀")
                    node.mode = "EXPLORE"
                    node.box_detected = False
                    continue
                
                if not node.is_navigating:
                    bx, by = node.current_box_pos
                    
                    # 박스 뒤 목표 지점 계산
                    pose = node.get_robot_pose()
                    if pose:
                        rx, ry, _ = pose
                        angle = math.atan2(by - ry, bx - rx)

                        # 박스 "뒤"로 이동
                        tx = bx + (BOX_DEPTH + BOX_BEHIND_OFFSET) * math.cos(angle)
                        ty = by + (BOX_DEPTH + BOX_BEHIND_OFFSET) * math.sin(angle)
                        
                        # 박스를 바라보도록
                        face_angle = angle + math.pi
                        qz = math.sin(face_angle / 2)
                        qw = math.cos(face_angle / 2)
                        
                        print(f"[APPROACH] 박스 뒤로 접근 시작!")
                        print(f"  현재 위치: ({rx:.2f}, {ry:.2f})")
                        print(f"  박스 앞면: ({bx:.2f}, {by:.2f})")
                        print(f"  목표 (박스 뒤): ({tx:.2f}, {ty:.2f})")

                        goal = PoseStamped()
                        goal.header.frame_id = 'map'
                        goal.header.stamp = node.nav.get_clock().now().to_msg()
                        goal.pose.position.x = tx
                        goal.pose.position.y = ty
                        goal.pose.orientation.z = qz
                        goal.pose.orientation.w = qw
                        
                        # Nav2 goal 전송
                        node.nav.goToPose(goal)
                        print(f"[APPROACH] Nav2 goal 전송 완료!")

                        node.nav.cancelTask()
                        node.target_point_pub.publish(goal)

                        node.is_navigating = True
                        node.box_detected = False
                    else:
                        print("로봇 위치 불명 -> 탐사 복귀")
                        node.mode = "EXPLORE"
                        node.box_detected = False
                
                elif node.nav.isTaskComplete():
                    result = node.nav.getResult()
                    if result == TaskResult.SUCCEEDED:
                        print('도착착!!!!!')
                        break

                        is_final_box = (len(node.checked_boxes) + 1) >= TARGET_BOX_COUNT
                        if is_final_box:
                            print("박스 도착 완료! 목표 수량 달성.")
                        else:
                            print("박스 도착 완료! (3초 대기)")
                            node.wait_async(3.0)
                        
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
                    with node.data_lock:
                        map_data = node.map_data
                        map_info = node.map_info
                        last_goal = node.last_goal
                    
                    target = compute_frontier_goal(map_data, map_info, last_goal)
                    if target:
                        tx, ty = target
                        print(f"\n탐사 목표: ({tx:.2f}, {ty:.2f})")
                        
                        goal = PoseStamped()
                        goal.header.frame_id = 'map'
                        goal.header.stamp = node.nav.get_clock().now().to_msg()
                        goal.pose.position.x = tx
                        goal.pose.position.y = ty
                        goal.pose.orientation.w = 1.0
                        
                        node.nav.goToPose(goal)
                        node.is_navigating = True
                        node.last_goal = (tx, ty)
                    else:
                        print("더 이상 갈 곳이 없음 (탐사 완료)")
                        node.wait_async(2.0)
                
                elif node.nav.isTaskComplete():
                    result = node.nav.getResult()
                    node.is_navigating = False
                    
                    if result == TaskResult.SUCCEEDED:
                        print("탐사 목표 도착!, 🔄 주변 박스 스캔 시작...")
                        
                        found = node.rotate_scan(duration_sec=10.0, angular_speed=0.16)
                        if found:
                            print("박스 발견! APPROACH 모드로 전환됨")
                            node.mode = "APPROACH"
                            with node.data_lock:
                                node.current_box_pos = node.fusion_box_world

                    else:
                        # 실패 시 바로 다음 목표로 (회전 스캔 생략)
                        print(f"탐사 목표 도달 실패: {result} → 다음 목표로")

    except KeyboardInterrupt:
        print("\n사용자 종료")

    if node.shutdown_requested:
        print("목표 박스 두 개 확보 완료. 노드를 종료합니다.")

    executor.shutdown()
    node.nav.lifecycleShutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
