#!/usr/bin/env python3
"""
Frontier 탐사 유틸 함수 모듈
"""

import math
import numpy as np
import cv2


def grid_to_world(map_info, gx, gy):
    """그리드 좌표 -> 월드 좌표 변환"""
    ox = map_info.origin.position.x
    oy = map_info.origin.position.y
    res = map_info.resolution
    return ox + gx * res, oy + gy * res


def find_safe_point(map_data, cx, cy):
    """주변 5픽셀 내에서 가장 안전한 Free 공간 찾기"""
    rows, cols = map_data.shape
    for r in range(max(0, cy - 5), min(rows, cy + 6)):
        for c in range(max(0, cx - 5), min(cols, cx + 6)):
            if map_data[r, c] == 0:
                # 너무 가까운(경계선) 곳은 피함 (2픽셀 이상)
                if abs(r - cy) + abs(c - cx) > 2:
                    return c, r  # (x, y)
    return None


def compute_frontier_goal(map_data, map_info, last_goal=None):
    """
    Frontier 기반 탐사 목표 계산
    반환: (wx, wy) 또는 None
    """
    if map_data is None or map_info is None:
        return None

    grid = map_data
    free_mask = (grid == 0).astype(np.uint8) * 255
    unknown_mask = (grid == -1).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    dilated_free = cv2.dilate(free_mask, kernel, iterations=1)
    frontier = cv2.bitwise_and(dilated_free, unknown_mask)
    contours, _ = cv2.findContours(frontier, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    candidates = []
    for cnt in contours:
        if len(cnt) < 5:
            continue

        m = cv2.moments(cnt)
        if m['m00'] == 0:
            continue
        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])

        safe_pt = find_safe_point(map_data, cx, cy)
        if not safe_pt:
            continue

        wx, wy = grid_to_world(map_info, *safe_pt)

        score = len(cnt)
        if last_goal:
            dist = math.hypot(wx - last_goal[0], wy - last_goal[1])
            if dist < 1.0:
                score *= 0.1  # 갔던 곳 회피

        candidates.append((score, wx, wy))

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1], candidates[0][2]

