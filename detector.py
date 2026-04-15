"""
화면 캡처 및 지뢰찾기 보드 상태 인식.

인식 모드:
  A) 레퍼런스 모드 (학습 완료 시):
     각 셀을 학습된 레퍼런스 이미지들과 MSE 비교 → 가장 유사한 유형으로 분류.
     공개됨(EMPTY)으로 분류된 셀은 색상 기반 숫자 감지.
  B) 베이스라인 모드 (학습 없이 시작 시, fallback):
     게임 시작 시 각 셀 초기 이미지 저장 → 변화 여부로 판정.

셀 상태값:
  -2 = UNKNOWN (미확인)
  -1 = FLAG (깃발)
  -3 = MINE (지뢰 터짐, 게임 오버)
   0 = EMPTY (빈 칸)
  1~8 = 숫자
"""

import os
import json
import numpy as np
import pyautogui
from PIL import Image


# 표준 지뢰찾기 숫자 색상 (RGB)
NUMBER_COLORS = {
    1: np.array([0,   0,   255]),   # 파랑
    2: np.array([0,   128,  0 ]),   # 초록
    3: np.array([255,  0,   0 ]),   # 빨강
    4: np.array([0,   0,   128]),   # 남색
    5: np.array([128,  0,   0 ]),   # 마룬
    6: np.array([0,   128, 128]),   # 청록
    7: np.array([0,   0,   0  ]),   # 검정
    8: np.array([128, 128, 128]),   # 회색
}

UNKNOWN = -2
FLAG    = -1
MINE    = -3
EMPTY   =  0

REF_SIZE = (24, 24)

# 레퍼런스 카테고리 이름 ↔ 상태값
CAT_NAMES  = {UNKNOWN: 'unknown', EMPTY: 'revealed', FLAG: 'flag', MINE: 'mine'}
NAME_CATS  = {v: k for k, v in CAT_NAMES.items()}

# 저장 경로
PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profiles')


# ════════════════════════════════════════════════════════════════════
#  레퍼런스 저장 / 불러오기
# ════════════════════════════════════════════════════════════════════

def save_references(refs, name='default'):
    """refs: {state_int: [numpy_array, ...]}"""
    os.makedirs(PROFILE_DIR, exist_ok=True)
    path = os.path.join(PROFILE_DIR, f'{name}.npz')

    save_dict = {}
    meta = {}
    for state, imgs in refs.items():
        cat = CAT_NAMES.get(state, str(state))
        meta[cat] = len(imgs)
        for i, img in enumerate(imgs):
            save_dict[f'{cat}_{i}'] = img

    save_dict['_meta'] = json.dumps(meta)
    np.savez_compressed(path, **save_dict)
    print(f"  레퍼런스 저장: {path} ({sum(meta.values())}개)")


def load_references(name='default'):
    """저장된 레퍼런스 불러오기. 없으면 None."""
    path = os.path.join(PROFILE_DIR, f'{name}.npz')
    if not os.path.exists(path):
        return None

    data = np.load(path, allow_pickle=True)
    meta = json.loads(str(data['_meta']))

    refs = {}
    for cat_name, count in meta.items():
        if cat_name in NAME_CATS:
            state = NAME_CATS[cat_name]
        else:
            state = int(cat_name)
        refs[state] = [data[f'{cat_name}_{i}'] for i in range(count)]

    total = sum(len(v) for v in refs.values())
    print(f"  레퍼런스 로드: {path} ({total}개)")
    return refs


def list_profiles():
    """저장된 프로필 이름 목록."""
    if not os.path.isdir(PROFILE_DIR):
        return []
    return [f[:-4] for f in os.listdir(PROFILE_DIR) if f.endswith('.npz')]


# ════════════════════════════════════════════════════════════════════
#  BoardDetector
# ════════════════════════════════════════════════════════════════════

class BoardDetector:
    def __init__(self, region, rows, cols, references=None, threshold=40):
        """
        region     : (x, y, w, h) — 화면에서의 보드 영역
        rows/cols  : 그리드 크기
        references : 학습된 레퍼런스 {state: [numpy_24x24, ...]} 또는 None
        """
        self.region    = region
        self.rows      = rows
        self.cols      = cols
        self.cell_w    = region[2] / cols
        self.cell_h    = region[3] / rows
        self.threshold = threshold

        # 레퍼런스 모드
        self._refs     = references
        self._use_refs = references is not None and len(references) > 0

        # 상태 잠금
        self._locked       = [[False] * cols for _ in range(rows)]
        self._locked_state = [[UNKNOWN] * cols for _ in range(rows)]

        # 베이스라인 (항상 준비, fallback + 변화 감지용)
        self._baselines    = None
        self._change_thresh = 40
        self._calibrate()

        mode = "레퍼런스" if self._use_refs else "베이스라인"
        print(f"  인식 모드: {mode}")

    # ------------------------------------------------------------------ #
    #  초기화
    # ------------------------------------------------------------------ #

    def _calibrate(self):
        """게임 시작 시 각 셀의 이미지를 개별 저장 (베이스라인)."""
        img = self._screenshot()
        self._baselines = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                x1, y1, x2, y2 = self._bounds(r, c)
                cell = img[y1:y2, x1:x2].astype(np.float32)
                row.append(cell)
            self._baselines.append(row)

        # 자연 편차 → 임계값 자동 보정
        mses = []
        for r in range(min(3, self.rows)):
            for c in range(min(self.cols - 1, 5)):
                a = self._baselines[r][c]
                b = self._baselines[r][c + 1]
                if a.shape == b.shape:
                    mses.append(float(np.mean((a - b) ** 2)))
        if mses:
            self._change_thresh = max(40, min(300, np.median(mses) * 2 + 20))

    # ------------------------------------------------------------------ #
    #  공개 메서드
    # ------------------------------------------------------------------ #

    def cell_center(self, row, col):
        x = self.region[0] + (col + 0.5) * self.cell_w
        y = self.region[1] + (row + 0.5) * self.cell_h
        return int(x), int(y)

    def capture_board(self):
        img = self._screenshot()
        board = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if self._locked[r][c]:
                    row.append(self._locked_state[r][c])
                    continue

                state = self._identify_cell(img, r, c)

                if state not in (UNKNOWN, FLAG, MINE):
                    self._locked[r][c] = True
                    self._locked_state[r][c] = state

                row.append(state)
            board.append(row)
        return board

    @staticmethod
    def capture_region(region):
        """영역 스크린샷 (학습 모드용)."""
        return np.array(pyautogui.screenshot(region=region))

    def get_cell_ref(self, img, row, col):
        """학습 모드용: 셀 이미지를 레퍼런스 크기로 정규화해서 반환."""
        x1, y1, x2, y2 = self._bounds(row, col)
        cell = img[y1:y2, x1:x2]
        if cell.size == 0 or cell.shape[0] < 2 or cell.shape[1] < 2:
            return None
        pil = Image.fromarray(cell)
        return np.array(pil.resize(REF_SIZE, Image.LANCZOS))

    # ------------------------------------------------------------------ #
    #  내부: 분류
    # ------------------------------------------------------------------ #

    def _identify_cell(self, img, row, col):
        if self._use_refs:
            return self._classify_by_ref(img, row, col)
        return self._classify_by_baseline(img, row, col)

    def _classify_by_ref(self, img, row, col):
        """레퍼런스 기반 분류."""
        norm = self.get_cell_ref(img, row, col)
        if norm is None:
            return UNKNOWN

        norm_f = norm.astype(np.float32)
        best_state = UNKNOWN
        best_mse   = float('inf')

        for state, ref_list in self._refs.items():
            for ref_img in ref_list:
                mse = float(np.mean((norm_f - ref_img.astype(np.float32)) ** 2))
                if mse < best_mse:
                    best_mse   = mse
                    best_state = state

        interior = self._interior(img, row, col)

        if best_state == EMPTY:
            # 공개된 셀 → 숫자 세부 감지
            if interior.size > 0:
                return self._detect_number(interior)

        elif best_state == MINE:
            # 레퍼런스가 MINE으로 분류했더라도 실제 지뢰처럼 보이지 않으면
            # 숫자로 오인식된 케이스 (특히 5번: 마룬 r=128, 임계값 r>150 불통과)
            # _is_mine_icon: 빨간 배경(red_ratio>0.01) + 어두운 아이콘(dark_ratio>0.12)
            # 숫자 5 셀은 red_ratio≈0 → False → 숫자 감지로 폴백
            if interior.size > 0 and not self._is_mine_icon(interior):
                detail = self._detect_number(interior)
                if 1 <= detail <= 8:
                    return detail

        return best_state

    def _classify_by_baseline(self, img, row, col):
        """베이스라인 기반 분류 (fallback)."""
        x1, y1, x2, y2 = self._bounds(row, col)
        current  = img[y1:y2, x1:x2].astype(np.float32)
        baseline = self._baselines[row][col]
        if current.shape != baseline.shape:
            return UNKNOWN

        mse = float(np.mean((current - baseline) ** 2))
        if mse < self._change_thresh:
            return UNKNOWN

        # 변화 있음 → 내용 분석
        interior = self._interior(img, row, col)
        if interior.size == 0:
            return UNKNOWN

        # 깃발
        if self._is_flag(interior):
            return FLAG

        return self._detect_number(interior)

    # ------------------------------------------------------------------ #
    #  내부: 헬퍼
    # ------------------------------------------------------------------ #

    def _screenshot(self):
        return np.array(pyautogui.screenshot(region=self.region))

    def _bounds(self, row, col):
        x1 = int(col * self.cell_w)
        y1 = int(row * self.cell_h)
        x2 = int((col + 1) * self.cell_w)
        y2 = int((row + 1) * self.cell_h)
        return x1, y1, x2, y2

    def _interior(self, img, row, col, margin_ratio=0.25):
        x1, y1, x2, y2 = self._bounds(row, col)
        m = max(1, int(min(x2 - x1, y2 - y1) * margin_ratio))
        return img[y1 + m:y2 - m, x1 + m:x2 - m]

    def _is_flag(self, interior):
        r = interior[:, :, 0].astype(float)
        g = interior[:, :, 1].astype(float)
        b = interior[:, :, 2].astype(float)
        red_mask = (r > 180) & (g < 90) & (b < 90)
        total = interior.shape[0] * interior.shape[1]
        return total > 0 and (red_mask.sum() / total) > 0.03

    def _detect_number(self, cell_img):
        if cell_img.size == 0:
            return EMPTY

        r = cell_img[:, :, 0].astype(float)
        g = cell_img[:, :, 1].astype(float)
        b = cell_img[:, :, 2].astype(float)

        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        sat = max_c - min_c

        colored = sat > 35
        if colored.sum() < 4:
            # 지뢰 감지는 보수적으로 처리 (숫자 2 등 오인식 방지)
            if self._is_mine_icon(cell_img):
                return MINE
            return EMPTY

        avg = np.array([r[colored].mean(), g[colored].mean(), b[colored].mean()])

        best_num  = EMPTY
        best_dist = 18000

        for num, ref in NUMBER_COLORS.items():
            d = float(((avg - ref) ** 2).sum())
            if d < best_dist:
                best_dist = d
                best_num  = num

        return best_num

    def _is_mine_icon(self, cell_img):
        if cell_img.size == 0:
            return False
        gray = cell_img.mean(axis=2)
        dark_ratio = (gray < 60).sum() / gray.size

        r = cell_img[:, :, 0].astype(float)
        g = cell_img[:, :, 1].astype(float)
        b = cell_img[:, :, 2].astype(float)
        red_ratio = ((r > 150) & (g < 110) & (b < 110)).sum() / gray.size

        return dark_ratio > 0.12 and red_ratio > 0.01
