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
BOX_CLASS_NAME = "box"       # YOLO 클래스 이름
BOX_APPROACH_DIST = 0.8      # 박스 앞 정지 거리 (m)
CHECKED_BOX_RADIUS = 1.5     # 이미 검사한 박스 반경 (m)
IMG_WIDTH = 320              # YOLO 입력 해상도 (px)
CENTER_TOLERANCE = 40        # 중앙 정렬 허용 오차 (px)
YOLO_CONF_THRESHOLD = 0.6    # YOLO 신뢰도 임계값
ALIGN_TIMEOUT = 10.0         # 정렬 타임아웃 (초) - 넉넉하게
ALIGN_LOST_COUNT = 10        # 박스 미감지 허용 횟수
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
        self.last_box_cx = IMG_WIDTH / 2  # 마지막 박스 X 좌표
        self.box_visible = False          # 현재 박스 보이는지
        self.box_lost_count = 0           # 박스 놓친 횟수
        
        # 7. TF (위치 추적용)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.get_logger().info("최종 탐사 노드 시작")

    # ===== 콜백 함수 =====
    def map_callback(self, msg):
        self.map_info = msg.info
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))

    def scan_callback(self, msg):
        """전방 거리 측정 (박스 위치 추정용)"""
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
        """박스 감지 시 바로 추적 접근"""
        try:
            detections = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        
        # 가장 신뢰도 높은 박스 찾기
        best_box = None
        best_conf = YOLO_CONF_THRESHOLD
        for det in detections:
            name = det.get('name', '').lower()
            conf = det.get('conf', 0.0)
            if name == BOX_CLASS_NAME.lower() and conf >= best_conf:
                best_conf = conf
                best_box = det
        
        # 박스 중앙 X 좌표 저장 (DIRECT_APPROACH에서 추적용)
        if best_box:
            center = best_box.get('center', [IMG_WIDTH / 2, 0.0])
            self.last_box_cx = center[0]
            self.box_visible = True
            self.box_lost_count = 0
        else:
            self.box_visible = False
            self.box_lost_count = getattr(self, 'box_lost_count', 0) + 1
        
        # 이미 접근 중이면 추적 정보만 업데이트하고 리턴
        if self.mode in ("APPROACH", "DIRECT_APPROACH"):
            return
        
        if not best_box:
            return
        
        # === 이미 확인한 박스인지 체크 ===
        temp_pos = self.estimate_box_position()
        if temp_pos and self.is_checked_box(*temp_pos):
            return  # 이미 간 박스는 무시
        
        # === 박스 발견 -> 바로 직접 접근 모드! ===
        self.get_logger().info(f"박스 발견! 바로 접근 시작 (cx={self.last_box_cx:.1f})")
        self.cancel_nav()
        self.box_detected = True
        self.mode = "DIRECT_APPROACH"
        
        # 현재 위치 저장 (나중에 박스 위치로 기록)
        pose = self.get_robot_pose()
        if pose:
            # 대략적인 박스 위치 추정
            dist = min(self.front_distance, 3.0)
            bx = pose[0] + dist * math.cos(pose[2])
            by = pose[1] + dist * math.sin(pose[2])
            self.current_box_pos = (bx, by)

    # ===== 유틸리티 =====
    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            q = t.transform.rotation
            yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))
            return t.transform.translation.x, t.transform.translation.y, yaw
        except Exception:
            return None

    def estimate_box_position(self):
        """로봇 정면 LiDAR 거리 기반 박스 위치 추정"""
        pose = self.get_robot_pose()
        if not pose:
            self.get_logger().warn("TF 조회 실패 - 로봇 위치 모름")
            return None
        rx, ry, ryaw = pose
        dist = min(self.front_distance, 3.0)  # 최대 3m로 제한
        bx = rx + dist * math.cos(ryaw)
        by = ry + dist * math.sin(ryaw)
        self.get_logger().info(f"박스 추정: 거리={dist:.2f}m, 위치=({bx:.2f}, {by:.2f})")
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
        # 정지 명령을 여러 번 보내서 확실히 멈춤
        for _ in range(3):
            self.publish_cmd_vel(0.0, 0.0)
        self.get_logger().info("Nav2 취소 요청")
    
    def publish_cmd_vel(self, linear_x, angular_z):
        """cmd_vel 퍼블리시"""
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(msg)
    
    def stop_robot(self):
        """로봇 정지"""
        self.publish_cmd_vel(0.0, 0.0)

    def wait_with_spin(self, duration_sec):
        """spin을 유지하면서 대기 (콜백 처리 계속)"""
        end_time = time.time() + duration_sec
        while time.time() < end_time and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.05)  # CPU 과부하 방지

    # ===== Frontier 로직 =====
    def get_frontier_point(self):
        if self.map_data is None:
            return None
        
        # 1. 마스크 생성
        grid = self.map_data
        free_mask = (grid == 0).astype(np.uint8) * 255
        unknown_mask = (grid == -1).astype(np.uint8) * 255
        
        # 2. Frontier 검출
        kernel = np.ones((3, 3), np.uint8)
        dilated_free = cv2.dilate(free_mask, kernel, iterations=1)
        frontier = cv2.bitwise_and(dilated_free, unknown_mask)
        contours, _ = cv2.findContours(frontier, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 3. 후보 선정
        candidates = []
        for cnt in contours:
            if len(cnt) < 5:
                continue  # 노이즈 제거
            
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
            
            wx, wy = self.grid_to_world(*safe_pt)
            
            # 점수: 크기(len) - 이전목표거리페널티
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
            
            # === 직접 직진 모드 (DIRECT_APPROACH) ===
            # Nav2 없이 직접 cmd_vel로 박스에 접근 (추적하며 직진)
            if node.mode == "DIRECT_APPROACH":
                dist = node.front_distance
                
                # 박스 놓친 경우 체크
                if node.box_lost_count > 30:  # 약 3초간 못 봄
                    print("박스 놓침 -> 탐사 복귀")
                    node.stop_robot()
                    node.mode = "EXPLORE"
                    node.box_detected = False
                    node.box_lost_count = 0
                    continue
                
                # 도착 체크
                if dist <= BOX_APPROACH_DIST:
                    # 박스 앞 도착!
                    node.stop_robot()
                    print(f"박스 앞 도착! (거리: {dist:.2f}m)")
                    node.wait_with_spin(3.0)
                    
                    # 현재 위치를 박스 위치로 기록
                    pose = node.get_robot_pose()
                    if pose:
                        node.checked_boxes.append((pose[0], pose[1]))
                        print(f"박스 위치 기록 (총 {len(node.checked_boxes)}개)")
                    
                    node.mode = "EXPLORE"
                    node.box_detected = False
                    node.box_lost_count = 0
                else:
                    # 추적하며 직진!
                    # 박스 X 위치에 따라 회전 보정
                    err = (IMG_WIDTH / 2) - node.last_box_cx
                    angular_z = 0.006 * err  # P 제어
                    angular_z = max(min(angular_z, 0.4), -0.4)
                    
                    # 직진 + 회전
                    linear_x = 0.15  # 전진 속도
                    node.publish_cmd_vel(linear_x, angular_z)
                    
                    if node.box_visible:
                        print(f"추적 직진: 거리={dist:.2f}m, 오차={err:.0f}px, 회전={angular_z:.2f}")
                    else:
                        print(f"박스 안 보임 (직진 유지): 거리={dist:.2f}m")
                    
                    time.sleep(0.1)
                continue
            
            # === Nav2 접근 모드 (APPROACH) ===
            if node.mode == "APPROACH":
                if node.current_box_pos is None:
                    # 위치 모르면 직접 직진 모드로!
                    print("박스 위치 없음 -> 직접 직진 모드")
                    node.mode = "DIRECT_APPROACH"
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
                        
                        print(f"박스 접근: ({tx:.2f}, {ty:.2f}), 방향: {math.degrees(angle):.1f} deg")
                        
                        goal = PoseStamped()
                        goal.header.frame_id = 'map'
                        goal.header.stamp = nav.get_clock().now().to_msg()
                        goal.pose.position.x = tx
                        goal.pose.position.y = ty
                        goal.pose.orientation.z = qz
                        goal.pose.orientation.w = qw
                        
                        nav.goToPose(goal)
                        node.is_navigating = True
                        node.box_detected = False
                    else:
                        # 위치 모르면 직접 직진!
                        print("로봇 위치 불명 -> 직접 직진 모드")
                        node.mode = "DIRECT_APPROACH"
                
                elif nav.isTaskComplete():
                    result = nav.getResult()
                    if result == TaskResult.SUCCEEDED:
                        print("박스 도착 완료! (3초 대기)")
                        node.wait_with_spin(3.0)  # spin 유지하면서 대기
                        
                        # 완료 처리
                        node.checked_boxes.append(node.current_box_pos)
                        print(f"박스 기록 완료 (총 {len(node.checked_boxes)}개)")
                    else:
                        print(f"박스 접근 실패: {result}")
                    
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

    except KeyboardInterrupt:
        print("\n사용자 종료")

    nav.lifecycleShutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
