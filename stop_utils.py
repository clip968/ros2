#!/usr/bin/env python3
"""
정지 로직 헬퍼 모듈
- cmd_vel 0,0을 짧게 반복 발행하며, 강제 정지 유지 시간 동안도 덮어쓰기
- 메인 루프에서 enforce_stop()를 주기적으로 호출해 Nav2 등의 명령을 무력화
"""

import time
import rclpy
from geometry_msgs.msg import Twist


class CmdStopper:
    def __init__(self, publisher, spin_node=None):
        """
        publisher : rclpy.Publisher(Twist) - cmd_vel 퍼블리셔
        spin_node : 콜백 처리 유지가 필요할 때 사용하는 rclpy.node.Node (옵션)
        """
        self.publisher = publisher
        self.spin_node = spin_node
        self.force_stop_until = 0.0

    def _publish_zero(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.publisher.publish(msg)

    def stop_now(self, duration_sec=0.5):
        """
        즉시 정지 명령을 duration_sec 동안 연속 발행.
        Nav2/타 노드의 cmd_vel을 덮어쓰고, 필요 시 spin_once로 콜백도 돌림.
        """
        end_time = time.time() + duration_sec
        self.force_stop_until = max(self.force_stop_until, end_time)
        while time.time() < end_time and rclpy.ok():
            self._publish_zero()
            if self.spin_node:
                rclpy.spin_once(self.spin_node, timeout_sec=0.0)
            time.sleep(0.02)  # 20ms 간격

    def enforce_stop(self):
        """
        메인 루프에서 주기적으로 호출.
        force_stop_until 이전이면 0,0을 발행하여 다른 명령을 차단.
        반환값: True(정지 강제 중), False(정지 필요 없음)
        """
        if time.time() < self.force_stop_until:
            self._publish_zero()
            if self.spin_node:
                rclpy.spin_once(self.spin_node, timeout_sec=0.0)
            return True
        return False

