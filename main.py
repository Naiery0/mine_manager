"""
지뢰찾기 자동 풀이 프로그램
사용법: python main.py [--debug]

  --debug  : 매 캡처마다 인식 결과를 이미지로 저장 (debug_capture.png)
"""

import sys
import time

import pyautogui

from region_selector import RegionSelector
from detector import BoardDetector, UNKNOWN, FLAG, MINE
from solver import MinesweeperSolver


# pyautogui 안전장치: 마우스를 화면 좌상단으로 이동하면 즉시 중단
pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0.03


# ------------------------------------------------------------------ #
#  유틸리티
# ------------------------------------------------------------------ #

def count_cells(board, state):
    return sum(1 for row in board for cell in row if cell == state)

def is_game_over(board):
    return any(cell == MINE for row in board for cell in row)

def is_game_won(board, rows, cols, mines):
    unknown = count_cells(board, UNKNOWN)
    flags   = count_cells(board, FLAG)
    return unknown + flags == mines


def get_grid_config():
    presets = {
        "1": ("초급",  9,  9,  10),
        "2": ("중급", 16, 16,  40),
        "3": ("고급",  16, 30,  99),
    }
    print()
    print("난이도를 선택하세요:")
    print("  1. 초급  ( 9×9,  지뢰 10)")
    print("  2. 중급  (16×16, 지뢰 40)")
    print("  3. 고급  (16×30, 지뢰 99)")
    print("  4. 직접 입력")

    ch = input("선택 (1-4): ").strip()
    if ch in presets:
        name, rows, cols, mines = presets[ch]
        print(f"  → {name} 선택")
        return rows, cols, mines

    rows  = int(input("세로 칸 수: ").strip())
    cols  = int(input("가로 칸 수: ").strip())
    mines = int(input("지뢰 수:   ").strip())
    return rows, cols, mines


# ------------------------------------------------------------------ #
#  메인
# ------------------------------------------------------------------ #

def main():
    debug = "--debug" in sys.argv

    print("=" * 52)
    print("    지뢰찾기 자동 풀이 프로그램")
    print("=" * 52)
    print("마우스를 화면 좌상단 모서리로 이동하면 즉시 종료됩니다.")

    # 1) 난이도 / 크기 입력
    rows, cols, mines = get_grid_config()

    # 2) 게임 영역 선택
    print()
    print("다음 창에서 게임 보드를 드래그해서 선택하세요...")
    time.sleep(0.5)

    selector = RegionSelector()
    region   = selector.select()

    if not region or region[2] < 30 or region[3] < 30:
        print("영역 선택이 취소되었습니다.")
        return

    print(f"선택 영역: x={region[0]}, y={region[1]}, "
          f"너비={region[2]}, 높이={region[3]}")

    # 3) 초기화
    print("\n보드 분석 중...")
    detector = BoardDetector(region, rows, cols)
    solver   = MinesweeperSolver(rows, cols, mines)

    print(f"그리드: {rows}행 × {cols}열  |  지뢰: {mines}개")
    print("풀이를 시작합니다. (중단: 마우스 좌상단 이동)\n")
    time.sleep(0.8)

    move_count   = 0
    stall_count  = 0
    prev_unknown = rows * cols

    while True:
        # ---- 캡처 ----
        if debug:
            board, debug_img = detector.capture_debug()
            debug_img.save("debug_capture.png")
        else:
            board = detector.capture_board()

        # ---- 종료 조건 ----
        if is_game_over(board):
            print(f"\n지뢰를 밟았습니다. ({move_count}번 이동)")
            break

        if is_game_won(board, rows, cols, mines):
            print(f"\n클리어! ({move_count}번 이동)")
            break

        # ---- 풀이 ----
        solver.update(board)
        safe_cells, mine_cells = solver.solve()

        if not safe_cells and not mine_cells:
            guess = solver.best_guess()
            if guess is None:
                print("더 이상 풀 수 없습니다.")
                break
            print(f"  [추측] ({guess[0]},{guess[1]})")
            safe_cells = [guess]
        else:
            if mine_cells:
                print(f"  [확정] 지뢰 {len(mine_cells)}개 표시, "
                      f"안전 {len(safe_cells)}개 클릭")

        # ---- 지뢰 깃발 표시 (우클릭) ----
        for r, c in mine_cells:
            if board[r][c] == UNKNOWN:
                x, y = detector.cell_center(r, c)
                pyautogui.rightClick(x, y)
                move_count += 1
                time.sleep(0.04)

        # ---- 안전 셀 클릭 ----
        for r, c in safe_cells:
            if board[r][c] == UNKNOWN:
                x, y = detector.cell_center(r, c)
                pyautogui.click(x, y)
                move_count += 1
                time.sleep(0.08)

        # ---- 애니메이션 대기 ----
        time.sleep(0.35)

        # ---- 교착 감지 ----
        new_board   = detector.capture_board()
        new_unknown = count_cells(new_board, UNKNOWN)

        if new_unknown >= prev_unknown and not mine_cells:
            stall_count += 1
            if stall_count >= 3:
                print("\n보드가 변하지 않습니다. "
                      "영역/크기 설정을 확인해주세요.")
                break
        else:
            stall_count = 0

        prev_unknown = new_unknown

    input("\nEnter를 누르면 종료...")


if __name__ == "__main__":
    main()
