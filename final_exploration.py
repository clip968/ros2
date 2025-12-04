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
from std_msgs.msg import String
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from tf2_ros import Buffer, TransformListener
import numpy as np
import cv2
import time
import math
import json

# ================= [설정] =================
BOX_CLASS_NAME = "box"       # YOLO 클래스 이름 (모델에 맞게 수정)
BOX_APPROACH_DIST = 0.6      # 박스 앞 정지 거리 (m)
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
        
        # 2. 퍼블리셔
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
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
        
        # 7. TF (위치 추적용)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.get_logger().info("최종 탐사 노드 시작")

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
        
        # === 이미 확인한 박스인지 체크 ===
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
        """라이다 데이터를 사용해 특정 각도의 거리를 구함"""
        if self.last_scan is None:
            self.get_logger().warn("LiDAR 데이터 없음 - 거리 추정 실패")
            return None
        scan = self.last_scan
        angle_min = scan.angle_min
        angle_max = scan.angle_min + scan.angle_increment * (len(scan.ranges) - 1)
        
        self.get_logger().info(
            f"라이다 범위: {math.degrees(angle_min):.1f}~{math.degrees(angle_max):.1f}deg, "
            f"요청 각도: {math.degrees(angle_rad):.1f}deg"
        )
        
        if angle_rad < angle_min or angle_rad > angle_max:
            self.get_logger().warn(
                f"요청 각도({math.degrees(angle_rad):.1f}deg)가 스캔 범위를 벗어남! "
                f"라이다 범위: {math.degrees(angle_min):.1f}~{math.degrees(angle_max):.1f}deg"
            )
            # 정면 거리로 대체
            self.get_logger().info("정면 거리로 대체 시도...")
            return self.front_distance if self.front_distance < float('inf') else None
            
        index = int(round((angle_rad - angle_min) / scan.angle_increment))
        window = 2  # ±2 샘플 평균 -> 노이즈 완화
        start = max(0, index - window)
        end = min(len(scan.ranges), index + window + 1)
        valid = [
            dist for dist in scan.ranges[start:end]
            if scan.range_min < dist < scan.range_max
        ]
        if not valid:
            self.get_logger().warn("해당 각도에서 유효한 라이다 거리 없음 - 정면 거리로 대체")
            return self.front_distance if self.front_distance < float('inf') else None
        return min(valid)

    def estimate_box_position(self, detection):
        """YOLO가 준 각도 정보를 이용해 박스 월드 좌표 추정"""
        pose = self.get_robot_pose()
        if not pose:
            self.get_logger().warn("TF 조회 실패 - 로봇 위치 모름")
            return None

        angle_rad = detection.get('angle_rad')
        if angle_rad is None:
            self.get_logger().warn("YOLO 감지 데이터에 angle_rad 없음")
            return None

        rx, ry, ryaw = pose
        distance = detection.get('range_m')
        if distance is None:
            distance = self.get_distance_along_angle(angle_rad)
        if distance is None:
            self.get_logger().warn("박스 거리 추정 실패 (라이다/센서 데이터 부족)")
            return None

        distance = min(distance, 3.0)  # 안전을 위해 최대 3m 제한
        heading = ryaw + angle_rad
        bx = rx + distance * math.cos(heading)
        by = ry + distance * math.sin(heading)
        self.front_distance = distance
        self.get_logger().info(
            f"박스 추정: bearing={math.degrees(angle_rad):.1f}deg, 거리={distance:.2f}m, 위치=({bx:.2f}, {by:.2f})"
        )
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
    
    def stop_robot(self):
        """로봇 정지 - 여러 번 발행해서 확실히 멈춤"""
        self.get_logger().info("정지 명령 발행!")
        # 1초 동안 계속 정지 명령 발행 (collision_monitor 덮어쓰기)
        for _ in range(50):  # 50번 발행
            self.publish_cmd_vel(0.0, 0.0)
            time.sleep(0.02)  # 20ms 간격 = 총 1초

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
    def get_frontier_point(self):
        if self.map_data is None:
            return None
        
        # 1. 마스크 생성
        # occupiedgrid값 
        # 0 = free, -1 = unknown, 100 = occupied
        grid = self.map_data
        free_mask = (grid == 0).astype(np.uint8) * 255
        unknown_mask = (grid == -1).astype(np.uint8) * 255
        
        # 2. Frontier 검출
        kernel = np.ones((3, 3), np.uint8)
        # free 영역을 3x3 커널로 1픽셀 확장
        dilated_free = cv2.dilate(free_mask, kernel, iterations=1)
        # 확장된 free와 unknown 마스크를 비트워드 연산으로 결합
        # 겹치는 부분 = frontier
        frontier = cv2.bitwise_and(dilated_free, unknown_mask)
        contours, _ = cv2.findContours(frontier, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 3. 후보 선정
        candidates = []
        for cnt in contours:
            # 너무 작은 frontier 무시
            if len(cnt) < 5:
                continue
            
            # 중심점 찾기
            m = cv2.moments(cnt)
            if m['m00'] == 0:
                continue
            cx = int(m['m10'] / m['m00'])
            cy = int(m['m01'] / m['m00'])
            
            # 안전한 위치로 보정 (Safe Point)
            safe_pt = self.find_safe_point(cx, cy)
            if not safe_pt:
                continue
            
            # 그리드 좌표를 map 좌표로 변환
            wx, wy = self.grid_to_world(*safe_pt)
            
            # 점수: 크기(len) - 이전목표거리페널티
            # frontier가 클 수록 높은 점수(넓은 미탐사 영역)
            # 뭔 개소리인지 모르겠음
            score = len(cnt)
            if self.last_goal:
                dist = math.hypot(wx - self.last_goal[0], wy - self.last_goal[1])
                if dist < 1.0:
                    score *= 0.1  # 갔던 곳 회피
                
            candidates.append((score, wx, wy))
            
        if not candidates:
            return None
        
        # 점수순 정렬
        candidates.sort(reverse=True, key=lambda x: x[0])
        return candidates[0][1], candidates[0][2]

    def find_safe_point(self, cx, cy):
        """주변 5픽셀 내에서 가장 안전한 Free 공간 찾기"""
        rows, cols = self.map_data.shape
        for r in range(max(0, cy - 5), min(rows, cy + 6)):
            for c in range(max(0, cx - 5), min(cols, cx + 6)):
                if self.map_data[r, c] == 0:
                    # 너무 가까운(경계선) 곳은 피함 (2픽셀 이상)
                    if abs(r - cy) + abs(c - cx) > 2:
                        return c, r  # (x, y)
        return None

    def grid_to_world(self, gx, gy):
        """그리드 좌표 -> 월드 좌표 변환"""
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        res = self.map_info.resolution
        return ox + gx * res, oy + gy * res


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
                        tx = bx - BOX_APPROACH_DIST * math.cos(angle)
                        ty = by - BOX_APPROACH_DIST * math.sin(angle)
                        
                        # 박스를 바라보는 orientation 계산
                        qz = math.sin(angle / 2)
                        qw = math.cos(angle / 2)
                        
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
                    target = node.get_frontier_point()
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
