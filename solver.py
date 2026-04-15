"""
지뢰찾기 풀이 로직.

전략 (순서대로):
1. 기본 규칙: 숫자 == 남은 미확인 이웃 수  →  전부 지뢰
             숫자 == 이미 표시된 깃발 수     →  나머지 안전
2. 부분집합 제약: 두 제약의 차집합으로 새 확정 추론 (로컬 제약만, 크기 제한 있음)
3. 글로벌-로컬 교차: 전체 남은 지뢰 수 + 로컬 제약 → 나머지 셀 확정
4. 확률 추정: 결정 불가 시 지뢰 확률이 가장 낮은 셀 추측
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

                if remaining < 0:
                    continue  # 오인식으로 인한 비정상 상태
                if remaining == 0:
                    safe.update(unknown_nb)
                elif remaining == len(unknown_nb):
                    mines.update(unknown_nb)

        # 2단계: 로컬 부분집합 제약
        s2, m2 = self._subset_solve()
        safe  |= s2
        mines |= m2

        # 3단계: 글로벌-로컬 교차 추론
        # (글로벌 제약을 _subset_solve에 직접 넣으면 대형 frozenset 파생으로
        #  지수적 폭발이 발생하므로 별도 단계로 분리)
        s3, m3 = self._global_local_solve()
        safe  |= s3
        mines |= m3

        # 이미 아는 셀 제거
        safe  = {c for c in safe  if self.board[c[0]][c[1]] == self.UNKNOWN}
        mines = {c for c in mines if self.board[c[0]][c[1]] == self.UNKNOWN}

        # 모순 검출: 오인식 등으로 같은 셀이 safe/mine 양쪽에 있으면 제거
        contradiction = safe & mines
        if contradiction:
            safe  -= contradiction
            mines -= contradiction

        return list(safe), list(mines)

    def best_guess(self):
        """
        결정론적 이동이 없을 때 가장 안전한 셀 추측.
        반환: (row, col) 또는 None.

        우선순위:
        1. 로컬 제약이 있는 셀 중 지뢰 확률 최솟값
        2. 제약 없는 셀은 글로벌 평균 확률 사용 (제약 있는 셀보다 후순위)
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
            # 첫 클릭: 중앙 (모서리는 제약 수가 적어 정보 획득 불리)
            return (self.rows//2, self.cols//2)

        flags_placed    = sum(1 for r in range(self.rows)
                              for c in range(self.cols)
                              if self.board[r][c] == self.FLAG)
        remaining_mines = max(0, self.total_mines - flags_placed)
        global_prob     = remaining_mines / len(unknown) if unknown else 0.5

        constrained, unconstrained = {}, {}

        for r, c in unknown:
            local_probs = []
            for nr, nc in self._neighbors(r, c):
                num = self.board[nr][nc]
                if num <= 0:
                    continue
                nb_unknown = self._unknown_neighbors(nr, nc)
                nb_flags   = self._flag_neighbors(nr, nc)
                nb_remain  = num - len(nb_flags)
                if nb_remain < 0:
                    continue
                if nb_unknown:
                    local_probs.append(nb_remain / len(nb_unknown))

            if local_probs:
                constrained[(r, c)] = max(local_probs)
            else:
                unconstrained[(r, c)] = global_prob

        # constrained/unconstrained 구분 없이 확률 최솟값 선택.
        # 이전 "constrained 우선" 방식은 constrained 확률이 global_prob보다
        # 높을 때도 constrained를 선택해 불필요하게 mine에 근접하는 문제가 있었음.
        all_probs = {**unconstrained, **constrained}
        return min(all_probs, key=all_probs.get)

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
        """로컬 제약 목록 반환: (frozenset of unknown cells, mine count)."""
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
                flags  = len(self._flag_neighbors(r, c))
                remain = num - flags
                if cells and 0 <= remain <= len(cells):
                    result.append((cells, remain))
        return result

    def _subset_solve(self):
        """
        로컬 제약만 사용한 부분집합 추론.
        A ⊆ B → (B-A)의 지뢰 수 = count_B - count_A.
        파생 제약은 MAX_CELLS 이하인 것만 풀에 추가해 폭발 방지.

        매 반복마다 확정된 셀(safe/mine)을 기존 제약에서 제거하여
        새로운 부분집합 관계가 드러나도록 한다 (연쇄 추론).
        """
        MAX_CELLS = 25

        safe, mines = set(), set()
        constraints = self._get_constraints()

        changed = True
        while changed:
            changed = False

            # ── 확정 셀을 제약에서 제거 (연쇄 추론 핵심) ──
            if safe or mines:
                updated = []
                for cells, count in constraints:
                    new_cells = cells - safe - mines
                    new_count = count - len(cells & mines)
                    if not new_cells:
                        continue
                    if new_count == 0:
                        if new_cells - safe:
                            safe |= new_cells
                            changed = True
                        continue
                    if new_count == len(new_cells):
                        if new_cells - mines:
                            mines |= new_cells
                            changed = True
                        continue
                    if 0 < new_count < len(new_cells):
                        updated.append((frozenset(new_cells), new_count))
                constraints = updated
                if changed:
                    continue

            # ── 부분집합 추론 ──
            new_constrs = []
            for i in range(len(constraints)):
                a_cells, a_count = constraints[i]
                for j in range(i + 1, len(constraints)):
                    b_cells, b_count = constraints[j]

                    for small_c, small_n, big_c, big_n in [
                        (a_cells, a_count, b_cells, b_count),
                        (b_cells, b_count, a_cells, a_count),
                    ]:
                        if not (small_c < big_c):
                            continue
                        diff       = big_c - small_c
                        diff_count = big_n - small_n

                        if diff_count == 0:
                            if diff - safe:
                                safe |= diff
                                changed = True
                        elif diff_count == len(diff):
                            if diff - mines:
                                mines |= diff
                                changed = True
                        elif 0 < diff_count < len(diff) and len(diff) <= MAX_CELLS:
                            nc = (frozenset(diff), diff_count)
                            if nc not in constraints and nc not in new_constrs:
                                new_constrs.append(nc)

            constraints.extend(new_constrs)

        return safe, mines

    def _global_local_solve(self):
        """
        글로벌 제약 (전체 미확인 = 남은 지뢰) + 각 로컬 제약 교차 추론.

        글로벌 제약을 _subset_solve에 넣으면 수백 셀짜리 파생 frozenset이
        지수적으로 폭발하므로, 여기서 직접 O(rows*cols) 로 처리한다.

        핵심 규칙:
          all_unknown = local_cells ∪ rest
          global_remain = local_remain + rest_remain
          → rest_remain == 0         : rest 전체 안전
          → rest_remain == len(rest) : rest 전체 지뢰

        추가: 비겹침 제약 쌍도 교차하여 단일 제약으로 놓치는 추론 보완.
        """
        safe, mines = set(), set()

        all_unknown = frozenset(
            (r, c) for r in range(self.rows) for c in range(self.cols)
            if self.board[r][c] == self.UNKNOWN
        )
        if not all_unknown:
            return safe, mines

        flags_total   = sum(1 for r in range(self.rows) for c in range(self.cols)
                            if self.board[r][c] == self.FLAG)
        global_remain = self.total_mines - flags_total

        # 기본 글로벌 규칙
        if global_remain == 0:
            safe.update(all_unknown)
            return safe, mines
        if global_remain == len(all_unknown):
            mines.update(all_unknown)
            return safe, mines

        # 로컬 제약 수집
        local_constraints = []
        for r in range(self.rows):
            for c in range(self.cols):
                num = self.board[r][c]
                if num <= 0:
                    continue
                local_cells  = frozenset(self._unknown_neighbors(r, c))
                if not local_cells:
                    continue
                local_flags  = len(self._flag_neighbors(r, c))
                local_remain = num - local_flags
                if local_remain < 0 or local_remain > len(local_cells):
                    continue
                local_constraints.append((local_cells, local_remain))

        # 단일 제약 교차
        for local_cells, local_remain in local_constraints:
            rest        = all_unknown - local_cells
            rest_remain = global_remain - local_remain
            if not rest or rest_remain < 0 or rest_remain > len(rest):
                continue
            if rest_remain == 0:
                safe.update(rest)
            elif rest_remain == len(rest):
                mines.update(rest)

        # 비겹침 제약 쌍 교차
        for i in range(len(local_constraints)):
            ci, ni = local_constraints[i]
            for j in range(i + 1, len(local_constraints)):
                cj, nj = local_constraints[j]
                if ci & cj:
                    continue  # 겹치면 단순 합산 불가
                union_cells  = ci | cj
                union_remain = ni + nj
                rest         = all_unknown - union_cells
                rest_remain  = global_remain - union_remain
                if not rest or rest_remain < 0 or rest_remain > len(rest):
                    continue
                if rest_remain == 0:
                    safe.update(rest)
                elif rest_remain == len(rest):
                    mines.update(rest)

        return safe, mines
