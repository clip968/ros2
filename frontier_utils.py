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


def is_safe_cell(map_data, r, c, safe_margin=3):
    """해당 셀 주변 safe_margin 픽셀 내에 장애물(100)이 없는지 확인"""
    rows, cols = map_data.shape
    for dr in range(-safe_margin, safe_margin + 1):
        for dc in range(-safe_margin, safe_margin + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if map_data[nr, nc] == 100:  # 장애물
                    return False
    return True

def find_safe_point(map_data, cx, cy):
    """주변 10픽셀 내에서 장애물로부터 충분히 떨어진 Free 공간 찾기"""
    rows, cols = map_data.shape
    best_pt = None
    best_dist = 0
    
    # 더 넓은 범위 탐색 (5 -> 10픽셀)
    for r in range(max(0, cy - 10), min(rows, cy + 11)):
        for c in range(max(0, cx - 10), min(cols, cx + 11)):
            if map_data[r, c] == 0:  # Free cell
                # 장애물과 최소 3픽셀 이상 떨어져야 함
                if is_safe_cell(map_data, r, c, safe_margin=3):
                    dist = abs(r - cy) + abs(c - cx)
                    if dist > 3 and dist > best_dist:  # 경계선에서 떨어짐
                        best_dist = dist
                        best_pt = (c, r)
    
    return best_pt


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

