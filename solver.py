"""
지뢰찾기 풀이 로직.

전략 (순서대로):
1. 기본 규칙: 숫자 == 남은 미확인 이웃 수  →  전부 지뢰
             숫자 == 이미 표시된 깃발 수     →  나머지 안전
2. 부분집합 제약: 두 제약의 차집합으로 새 확정 추론
3. 확률 추정: 결정 불가 시 지뢰 확률이 가장 낮은 셀 추측
"""


class MinesweeperSolver:
    UNKNOWN = -2
    FLAG    = -1

    def __init__(self, rows, cols, total_mines):
        self.rows        = rows
        self.cols        = cols
        self.total_mines = total_mines
        self.board       = [[self.UNKNOWN]*cols for _ in range(rows)]

    # ------------------------------------------------------------------ #
    #  공개 메서드
    # ------------------------------------------------------------------ #

    def update(self, board_state):
        """감지된 보드 상태로 내부 상태 갱신."""
        for r in range(self.rows):
            for c in range(self.cols):
                self.board[r][c] = board_state[r][c]

    def solve(self):
        """
        한 라운드 풀이 수행.
        반환: (safe_cells, mine_cells) — 각각 (row, col) 리스트.
        """
        safe, mines = set(), set()

        # 1단계: 기본 규칙
        for r in range(self.rows):
            for c in range(self.cols):
                num = self.board[r][c]
                if num <= 0:
                    continue
                unknown_nb = self._unknown_neighbors(r, c)
                flag_nb    = self._flag_neighbors(r, c)
                remaining  = num - len(flag_nb)

                if remaining == 0:
                    safe.update(unknown_nb)
                elif remaining == len(unknown_nb):
                    mines.update(unknown_nb)

        # 2단계: 부분집합 제약
        s2, m2 = self._subset_solve()
        safe  |= s2
        mines |= m2

        # 이미 아는 셀 제거
        safe  = {c for c in safe  if self.board[c[0]][c[1]] == self.UNKNOWN}
        mines = {c for c in mines if self.board[c[0]][c[1]] == self.UNKNOWN}

        return list(safe), list(mines)

    def best_guess(self):
        """
        결정론적 이동이 없을 때 가장 안전한 셀 추측.
        반환: (row, col) 또는 None.
        """
        unknown = [(r,c) for r in range(self.rows)
                   for c in range(self.cols)
                   if self.board[r][c] == self.UNKNOWN]
        if not unknown:
            return None

        revealed = any(self.board[r][c] >= 0
                       for r in range(self.rows)
                       for c in range(self.cols))
        if not revealed:
            # 첫 클릭: 중앙 근처
            return (self.rows//2, self.cols//2)

        flags_placed   = sum(1 for r in range(self.rows)
                             for c in range(self.cols)
                             if self.board[r][c] == self.FLAG)
        remaining_mines = max(0, self.total_mines - flags_placed)

        cell_prob = {}
        for r, c in unknown:
            local_probs = []
            for nr, nc in self._neighbors(r, c):
                num = self.board[nr][nc]
                if num <= 0:
                    continue
                nb_unknown = self._unknown_neighbors(nr, nc)
                nb_flags   = self._flag_neighbors(nr, nc)
                nb_remain  = num - len(nb_flags)
                if nb_unknown:
                    local_probs.append(nb_remain / len(nb_unknown))

            if local_probs:
                cell_prob[(r,c)] = max(local_probs)
            else:
                # 숫자 셀과 접하지 않으면 전체 평균 확률
                cell_prob[(r,c)] = (remaining_mines / len(unknown)
                                    if unknown else 0.5)

        return min(cell_prob, key=cell_prob.get)

    # ------------------------------------------------------------------ #
    #  내부 헬퍼
    # ------------------------------------------------------------------ #

    def _neighbors(self, r, c):
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    yield nr, nc

    def _unknown_neighbors(self, r, c):
        return [(nr,nc) for nr,nc in self._neighbors(r,c)
                if self.board[nr][nc] == self.UNKNOWN]

    def _flag_neighbors(self, r, c):
        return [(nr,nc) for nr,nc in self._neighbors(r,c)
                if self.board[nr][nc] == self.FLAG]

    def _get_constraints(self):
        """(frozenset of unknown cells, mine count) 제약 목록 반환."""
        result = []
        for r in range(self.rows):
            for c in range(self.cols):
                num = self.board[r][c]
                if num <= 0:
                    continue
                cells = frozenset(
                    (nr,nc) for nr,nc in self._neighbors(r,c)
                    if self.board[nr][nc] == self.UNKNOWN
                )
                flags = len(self._flag_neighbors(r, c))
                remain = num - flags
                if cells and 0 <= remain <= len(cells):
                    result.append((cells, remain))
        return result

    def _subset_solve(self):
        """
        두 제약 A ⊆ B 일 때 (B-A)의 지뢰 수 = count_B - count_A 를 새 제약으로 추가.
        확정 결론이 나올 때까지 반복.
        """
        safe, mines = set(), set()
        constraints = self._get_constraints()

        changed = True
        while changed:
            changed      = False
            new_constrs  = []

            for i in range(len(constraints)):
                a_cells, a_count = constraints[i]
                for j in range(i+1, len(constraints)):
                    b_cells, b_count = constraints[j]

                    for small_c, small_n, big_c, big_n in [
                        (a_cells, a_count, b_cells, b_count),
                        (b_cells, b_count, a_cells, a_count),
                    ]:
                        if not (small_c < big_c):   # 진부분집합 확인
                            continue
                        diff       = big_c - small_c
                        diff_count = big_n - small_n
                        if not diff:
                            continue

                        if diff_count == 0:
                            new_safe = {x for x in diff if self.board[x[0]][x[1]] == self.UNKNOWN}
                            if new_safe - safe:
                                safe |= new_safe
                                changed = True
                        elif diff_count == len(diff):
                            new_m = {x for x in diff if self.board[x[0]][x[1]] == self.UNKNOWN}
                            if new_m - mines:
                                mines |= new_m
                                changed = True
                        elif 0 < diff_count < len(diff):
                            nc = (frozenset(diff), diff_count)
                            if nc not in constraints and nc not in new_constrs:
                                new_constrs.append(nc)

            constraints.extend(new_constrs)

        return safe, mines
