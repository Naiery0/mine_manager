"""
지뢰찾기 자동 풀이 GUI
실행: python app.py
"""

import tkinter as tk
import threading
import queue
import time

import numpy as np
import pyautogui
from PIL import Image, ImageTk, ImageDraw

from detector import (BoardDetector, UNKNOWN, FLAG, MINE, EMPTY,
                       REF_SIZE, CAT_NAMES, save_references, load_references)
from solver import MinesweeperSolver

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.02

# ── 폰트 ──────────────────────────────────────────────────────────────
FONT = 'Pretendard'

# ── 색상 팔레트 ───────────────────────────────────────────────────────
C = {
    'bg':        '#F5F6FA',   # 앱 배경
    'panel':     '#FFFFFF',   # 왼쪽 패널
    'header_bg': '#2563EB',   # 헤더 배경
    'header_fg': '#FFFFFF',   # 헤더 글자
    'sec_fg':    '#374151',   # 섹션 제목
    'divider':   '#E5E7EB',   # 구분선
    'body':      '#1F2937',   # 본문 글자
    'secondary': '#6B7280',   # 보조 글자
    'accent':    '#2563EB',   # 강조 (파란색)
    'warn':      '#D97706',   # 경고 (주황)
    'error':     '#DC2626',   # 오류 (빨강)
    'success':   '#16A34A',   # 성공 (초록)
    'input_bg':  '#F9FAFB',   # 입력창 배경
    'canvas_bg': '#F3F4F6',   # 캔버스 배경
    'log_bg':    '#F9FAFB',   # 로그 배경
    'radio_sel': '#DBEAFE',   # 라디오버튼 선택색
    'trough':    '#E5E7EB',   # 슬라이더 홈
}

# ── 보드 시각화 색상 ──────────────────────────────────────────────────
CELL_FILL = {
    UNKNOWN: '#CBD5E1',
    FLAG:    '#EF4444',
    MINE:    '#991B1B',
    EMPTY:   '#F1F5F9',
}
NUM_COLOR = {
    1: '#1D4ED8', 2: '#15803D', 3: '#DC2626',
    4: '#1E3A8A', 5: '#7C3AED', 6: '#0E7490',
    7: '#374151', 8: '#6B7280',
}
PRESETS = [
    ('초급',  9,  9,  10),
    ('중급', 16, 16,  40),
    ('고급',  16, 30,  99),
]

LEARN_LABELS = [
    (UNKNOWN, '미확인', '#CBD5E1'),
    (EMPTY,   '공개됨', '#F1F5F9'),
    (FLAG,    '깃발',  '#EF4444'),
    (MINE,    '지뢰',  '#991B1B'),
]


# ════════════════════════════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("지뢰찾기 자동 풀이")
        self.geometry("1050x700")
        self.minsize(820, 560)
        self.configure(bg=C['bg'])

        # 상태
        self.region      = None
        self.detector    = None
        self.solver      = None
        self.stop_evt    = threading.Event()
        self.board_q     = queue.Queue(maxsize=2)
        self.log_q       = queue.Queue()
        self._board_cache = None
        self._cell_ids    = None

        # 학습 모드 상태
        self._learning       = False
        self._learn_label    = UNKNOWN
        self._learn_refs     = {}      # {state: [numpy_24x24, ...]}
        self._learn_img      = None    # 캡처된 원본 이미지
        self._learn_labeled  = {}      # {(r,c): state} — 라벨링된 셀
        self._learn_display  = None    # (img_x0, img_y0, scale, rows, cols)

        # 저장된 레퍼런스
        self._saved_refs = load_references()

        self._build_ui()
        self._poll()
        self.bind('<Escape>', lambda _: self._on_escape())

    # ── UI 구성 ─────────────────────────────────────────────────────
    def _build_ui(self):
        # ── 왼쪽 패널 ──────────────────────────────
        L = tk.Frame(self, bg=C['panel'], width=260)
        L.pack(side='left', fill='y')
        L.pack_propagate(False)
        self._left_panel = L

        def section(text):
            tk.Label(L, text=text, bg=C['panel'], fg=C['sec_fg'],
                     font=(FONT, 9, 'bold')).pack(anchor='w', padx=14, pady=(12, 2))
            tk.Frame(L, height=1, bg=C['divider']).pack(fill='x', padx=10)

        # 헤더
        tk.Label(L, text="지뢰찾기 자동풀이", bg=C['header_bg'], fg=C['header_fg'],
                 font=(FONT, 13, 'bold'), pady=12).pack(fill='x')

        # 난이도
        section("난이도")
        self.diff_var = tk.StringVar(value='중급')
        for name, rows, cols, mines in PRESETS:
            tk.Radiobutton(L, text=f"{name}  {cols}×{rows}  지뢰{mines}",
                           variable=self.diff_var, value=name,
                           command=lambda r=rows, c=cols, m=mines: self._set_preset(r, c, m),
                           bg=C['panel'], fg=C['body'], selectcolor=C['radio_sel'],
                           activebackground=C['panel'], activeforeground=C['body'],
                           font=(FONT, 10)).pack(anchor='w', padx=14)

        # 커스텀 입력
        section("커스텀")
        grid = tk.Frame(L, bg=C['panel'])
        grid.pack(fill='x', padx=14, pady=4)
        self.rows_v  = tk.IntVar(value=16)
        self.cols_v  = tk.IntVar(value=16)
        self.mines_v = tk.IntVar(value=40)
        for i, (lbl, var) in enumerate([('행', self.rows_v), ('열', self.cols_v), ('지뢰', self.mines_v)]):
            tk.Label(grid, text=lbl + ':', bg=C['panel'], fg=C['secondary'],
                     font=(FONT, 10), width=4, anchor='w').grid(row=i, column=0, sticky='w', pady=2)
            tk.Entry(grid, textvariable=var, width=7,
                     bg=C['input_bg'], fg=C['body'], insertbackground=C['body'],
                     relief='solid', bd=1, font=(FONT, 10)).grid(row=i, column=1, sticky='w')

        # 속도
        section("속도")
        self.speed_v = tk.IntVar(value=200)
        row = tk.Frame(L, bg=C['panel'])
        row.pack(fill='x', padx=14, pady=4)
        tk.Label(row, text='속도', bg=C['panel'], fg=C['secondary'],
                 font=(FONT, 9), width=5, anchor='w').pack(side='left')
        val_lbl = tk.Label(row, text=f"{self.speed_v.get()}ms", bg=C['panel'],
                           fg=C['accent'], font=(FONT, 9, 'bold'), width=6)
        val_lbl.pack(side='right')
        tk.Scale(row, from_=30, to=800, variable=self.speed_v, orient='horizontal',
                 bg=C['panel'], fg=C['body'], troughcolor=C['trough'],
                 highlightthickness=0, showvalue=False,
                 command=lambda v, lbl=val_lbl: lbl.config(text=f"{int(float(v))}ms")
                 ).pack(side='left', fill='x', expand=True)

        # 영역 선택
        section("게임 영역")
        self.region_btn = self._flat_btn(L, "영역 선택 (2클릭)", self._select_region,
                                         bg='#374151', hover='#1F2937')
        self.region_btn.pack(fill='x', padx=10, pady=(8, 2))
        tk.Label(L, text="① 그리드 좌상단 클릭  ② 우하단 클릭",
                 bg=C['panel'], fg=C['warn'], font=(FONT, 8),
                 wraplength=230).pack(anchor='w', padx=14)
        self.region_lbl = tk.Label(L, text="선택 안 됨", fg=C['error'], bg=C['panel'],
                                    font=(FONT, 9), wraplength=230)
        self.region_lbl.pack(anchor='w', padx=14)
        self._flat_btn(L, "그리드 미리보기", self._preview_region,
                       bg='#6B7280', hover='#4B5563').pack(fill='x', padx=10, pady=(2, 0))

        # ── 학습 ──
        section("학습")
        self.learn_btn = self._flat_btn(L, "학습 모드", self._enter_learn_mode,
                                        bg='#0284C7', hover='#0369A1')
        self.learn_btn.pack(fill='x', padx=10, pady=(8, 2))

        # 학습 라벨 선택 (처음엔 숨김)
        self._learn_frame = tk.Frame(L, bg=C['panel'])
        self._learn_label_var = tk.IntVar(value=UNKNOWN)
        for state, text, color in LEARN_LABELS:
            f = tk.Frame(self._learn_frame, bg=C['panel'])
            f.pack(fill='x', padx=14, pady=1)
            tk.Radiobutton(f, text=text, variable=self._learn_label_var, value=state,
                           command=lambda s=state: self._set_learn_label(s),
                           bg=C['panel'], fg=C['body'], selectcolor=C['radio_sel'],
                           activebackground=C['panel'], activeforeground=C['body'],
                           font=(FONT, 10)).pack(side='left')
            tk.Frame(f, bg=color, width=12, height=12,
                     highlightbackground='#D1D5DB', highlightthickness=1).pack(side='right', padx=4)

        self.learn_count_lbl = tk.Label(self._learn_frame, text="", bg=C['panel'],
                                         fg=C['secondary'], font=(FONT, 8),
                                         justify='left', wraplength=220)
        self.learn_count_lbl.pack(anchor='w', padx=14, pady=4)

        self.learn_done_btn = self._flat_btn(self._learn_frame, "학습 완료", self._finish_learn,
                                             bg='#16A34A', hover='#15803D')
        self.learn_done_btn.pack(fill='x', padx=10, pady=2)
        self.learn_cancel_btn = self._flat_btn(self._learn_frame, "취소", self._cancel_learn,
                                               bg='#9CA3AF', hover='#6B7280')
        self.learn_cancel_btn.pack(fill='x', padx=10, pady=(0, 4))

        # 학습 상태
        self.learn_status_lbl = tk.Label(L, text="", fg=C['accent'], bg=C['panel'],
                                          font=(FONT, 8), wraplength=230)
        self.learn_status_lbl.pack(anchor='w', padx=14)
        self._update_learn_status()

        # 제어
        section("제어")
        br = tk.Frame(L, bg=C['panel'])
        br.pack(fill='x', padx=10, pady=8)
        self.start_btn = self._flat_btn(br, "시작", self._start,
                                        bg='#16A34A', hover='#15803D', state='disabled')
        self.start_btn.pack(side='left', fill='x', expand=True, padx=(0, 3))
        self.stop_btn  = self._flat_btn(br, "정지", self._stop,
                                        bg='#DC2626', hover='#B91C1C', state='disabled')
        self.stop_btn.pack(side='left', fill='x', expand=True)

        self.status_lbl = tk.Label(L, text="대기 중", fg=C['accent'], bg=C['panel'],
                                    font=(FONT, 9), justify='left')
        self.status_lbl.pack(anchor='w', padx=14, pady=4)

        # ── 오른쪽 패널 ─────────────────────────────
        R = tk.Frame(self, bg=C['bg'])
        R.pack(side='left', fill='both', expand=True)

        tk.Label(R, text="실시간 보드 인식", bg=C['bg'], fg=C['sec_fg'],
                 font=(FONT, 10, 'bold')).pack(anchor='w', padx=10, pady=(10, 4))

        self.canvas = tk.Canvas(R, bg=C['canvas_bg'], highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, padx=10)
        self.canvas.bind('<Configure>', lambda _: self._redraw())

        legend = tk.Frame(R, bg=C['bg'])
        legend.pack(fill='x', padx=10, pady=(4, 0))
        for color, text in [('#CBD5E1', '미확인'), ('#F1F5F9', '빈칸'),
                             ('#FFFFFF', '숫자'), ('#EF4444', '깃발'), ('#991B1B', '지뢰')]:
            tk.Frame(legend, bg=color, width=12, height=12,
                     highlightbackground='#D1D5DB', highlightthickness=1).pack(side='left', padx=(0, 2))
            tk.Label(legend, text=text, bg=C['bg'], fg=C['secondary'],
                     font=(FONT, 8)).pack(side='left', padx=(0, 10))

        tk.Label(R, text="로그", bg=C['bg'], fg=C['sec_fg'],
                 font=(FONT, 10, 'bold')).pack(anchor='w', padx=10, pady=(8, 4))

        lf = tk.Frame(R, bg=C['log_bg'], bd=1, relief='solid',
                      highlightbackground=C['divider'], highlightthickness=1)
        lf.pack(fill='x', padx=10, pady=(0, 10))
        self.log_txt = tk.Text(lf, height=5, state='disabled', bg=C['log_bg'], fg=C['secondary'],
                                font=(FONT, 9), wrap='word', relief='flat', bd=0)
        sb = tk.Scrollbar(lf, command=self.log_txt.yview, bg=C['bg'], troughcolor=C['trough'])
        self.log_txt.config(yscrollcommand=sb.set)
        self.log_txt.pack(side='left', fill='both', expand=True, padx=4, pady=4)
        sb.pack(side='right', fill='y')

    # ── 유틸: flat 버튼 ─────────────────────────────────────────────
    def _flat_btn(self, parent, text, cmd, bg, hover, state='normal'):
        btn = tk.Button(parent, text=text, command=cmd, state=state,
                        bg=bg, fg='white', activebackground=hover, activeforeground='white',
                        disabledforeground='#9CA3AF', relief='flat',
                        font=(FONT, 10, 'bold'), pady=7, bd=0, cursor='hand2')
        btn.bind('<Enter>', lambda _: btn.config(bg=hover) if btn['state'] == 'normal' else None)
        btn.bind('<Leave>', lambda _: btn.config(bg=bg))
        return btn

    def _set_preset(self, rows, cols, mines):
        self.rows_v.set(rows); self.cols_v.set(cols); self.mines_v.set(mines)

    def _on_escape(self):
        if self._learning:
            self._cancel_learn()
        else:
            self._stop()

    # ═══════════════════════════════════════════════════════════════
    #  학습 모드
    # ═══════════════════════════════════════════════════════════════

    def _update_learn_status(self):
        if self._saved_refs:
            parts = []
            for state, name, _ in LEARN_LABELS:
                cnt = len(self._saved_refs.get(state, []))
                if cnt > 0:
                    parts.append(f"{name} {cnt}")
            self.learn_status_lbl.config(
                text=f"학습 데이터: {', '.join(parts)}" if parts else "")
        else:
            self.learn_status_lbl.config(text="학습 데이터 없음")

    def _enter_learn_mode(self):
        if not self.region:
            self._log("영역을 먼저 선택하세요")
            return

        self._learning = True
        self._learn_refs = {}
        self._learn_labeled = {}
        self._learn_label = UNKNOWN
        self._learn_label_var.set(UNKNOWN)

        # 보드 캡처
        try:
            self._learn_img = np.array(pyautogui.screenshot(region=self.region))
        except Exception as e:
            self._log(f"캡처 오류: {e}")
            self._learning = False
            return

        # UI 전환
        self.learn_btn.config(state='disabled')
        self.start_btn.config(state='disabled')
        self.region_btn.config(state='disabled')
        self._learn_frame.pack(fill='x', after=self.learn_btn)

        # 캔버스에 보드 이미지 + 그리드 표시
        self._show_learn_canvas()

        # 캔버스 클릭 바인딩
        self.canvas.bind('<ButtonPress-1>', self._on_learn_click)
        self._log("학습 모드: 라벨을 선택하고 해당 셀을 클릭하세요")

    def _show_learn_canvas(self):
        """학습 모드 캔버스: 보드 이미지 + 그리드 + 라벨 표시."""
        self.canvas.delete('all')
        self._cell_ids = None

        img = self._learn_img
        rows = self.rows_v.get()
        cols = self.cols_v.get()
        sh, sw = img.shape[:2]

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 500, 350

        scale = min(cw / sw, ch / sh) * 0.95
        dw, dh = int(sw * scale), int(sh * scale)
        x0 = (cw - dw) // 2
        y0 = (ch - dh) // 2

        self._learn_display = (x0, y0, scale, rows, cols)

        # 이미지 표시
        pil = Image.fromarray(img).resize((dw, dh), Image.LANCZOS)
        self._learn_tk_img = ImageTk.PhotoImage(pil)
        self.canvas.create_image(x0, y0, image=self._learn_tk_img, anchor='nw', tags='bg')

        # 그리드 선
        cell_dw = dw / cols
        cell_dh = dh / rows
        for i in range(1, cols):
            x = x0 + int(i * cell_dw)
            self.canvas.create_line(x, y0, x, y0 + dh, fill='#2563EB', width=1, tags='grid')
        for i in range(1, rows):
            y = y0 + int(i * cell_dh)
            self.canvas.create_line(x0, y, x0 + dw, y, fill='#2563EB', width=1, tags='grid')
        self.canvas.create_rectangle(x0, y0, x0 + dw, y0 + dh, outline='#2563EB', width=2, tags='grid')

        # 라벨된 셀 표시
        self._redraw_learn_labels()

    def _redraw_learn_labels(self):
        self.canvas.delete('labels')
        if not self._learn_display:
            return
        x0, y0, scale, rows, cols = self._learn_display
        sw, sh = self._learn_img.shape[1], self._learn_img.shape[0]
        dw, dh = int(sw * scale), int(sh * scale)
        cell_dw = dw / cols
        cell_dh = dh / rows

        label_colors = {s: c for s, _, c in LEARN_LABELS}
        label_names  = {s: n for s, n, _ in LEARN_LABELS}

        for (r, c), state in self._learn_labeled.items():
            cx1 = x0 + int(c * cell_dw) + 1
            cy1 = y0 + int(r * cell_dh) + 1
            cx2 = x0 + int((c + 1) * cell_dw) - 1
            cy2 = y0 + int((r + 1) * cell_dh) - 1
            color = label_colors.get(state, '#ffffff')
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2,
                fill=color, outline='#ffffff', width=1,
                stipple='gray50', tags='labels')
            fs = max(6, int(min(cell_dw, cell_dh) * 0.3))
            self.canvas.create_text((cx1 + cx2) / 2, (cy1 + cy2) / 2,
                text=label_names.get(state, '?'), fill='white',
                font=(FONT, fs, 'bold'), tags='labels')

        # 카운트 업데이트
        parts = []
        for state, name, _ in LEARN_LABELS:
            cnt = len(self._learn_refs.get(state, []))
            if cnt > 0:
                parts.append(f"{name}: {cnt}")
        self.learn_count_lbl.config(text='\n'.join(parts) if parts else "셀을 클릭해 라벨링하세요")

    def _set_learn_label(self, state):
        self._learn_label = state

    def _on_learn_click(self, event):
        if not self._learning or not self._learn_display:
            return
        x0, y0, scale, rows, cols = self._learn_display
        sw, sh = self._learn_img.shape[1], self._learn_img.shape[0]
        dw, dh = int(sw * scale), int(sh * scale)

        # 캔버스 좌표 → 셀 (row, col)
        rx = event.x - x0
        ry = event.y - y0
        if rx < 0 or ry < 0 or rx >= dw or ry >= dh:
            return

        col = int(rx / (dw / cols))
        row = int(ry / (dh / rows))
        if row < 0 or row >= rows or col < 0 or col >= cols:
            return

        # 셀 이미지 추출 (원본 해상도에서)
        cell_w = sw / cols
        cell_h = sh / rows
        cx1 = int(col * cell_w)
        cy1 = int(row * cell_h)
        cx2 = int((col + 1) * cell_w)
        cy2 = int((row + 1) * cell_h)
        cell = self._learn_img[cy1:cy2, cx1:cx2]

        if cell.size == 0:
            return

        # REF_SIZE로 정규화
        pil = Image.fromarray(cell)
        norm = np.array(pil.resize(REF_SIZE, Image.LANCZOS))

        state = self._learn_label
        if state not in self._learn_refs:
            self._learn_refs[state] = []
        self._learn_refs[state].append(norm)
        self._learn_labeled[(row, col)] = state

        self._redraw_learn_labels()

    def _finish_learn(self):
        total = sum(len(v) for v in self._learn_refs.values())
        if total == 0:
            self._log("라벨링된 셀이 없습니다")
            return

        # 저장
        save_references(self._learn_refs)
        self._saved_refs = self._learn_refs.copy()
        self._update_learn_status()

        parts = []
        for state, name, _ in LEARN_LABELS:
            cnt = len(self._learn_refs.get(state, []))
            if cnt > 0:
                parts.append(f"{name} {cnt}")
        self._log(f"학습 완료: {', '.join(parts)}")

        self._exit_learn_mode()

    def _cancel_learn(self):
        self._log("학습 취소")
        self._exit_learn_mode()

    def _exit_learn_mode(self):
        self._learning = False
        self._learn_frame.pack_forget()
        self.canvas.unbind('<ButtonPress-1>')
        self.learn_btn.config(state='normal')
        self.region_btn.config(state='normal')
        if self.region:
            self.start_btn.config(state='normal')
        self.canvas.delete('all')
        self._cell_ids = None

    # ═══════════════════════════════════════════════════════════════
    #  영역 선택 (2클릭 방식)
    # ═══════════════════════════════════════════════════════════════

    def _select_region(self):
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
        phys_w = ctypes.windll.user32.GetSystemMetrics(0)
        phys_h = ctypes.windll.user32.GetSystemMetrics(1)
        log_w  = self.winfo_screenwidth()
        log_h  = self.winfo_screenheight()
        dpi_x  = phys_w / log_w
        dpi_y  = phys_h / log_h

        self.withdraw()
        self.update_idletasks()
        time.sleep(0.35)

        overlay = tk.Toplevel(self)
        overlay.overrideredirect(True)
        overlay.geometry(f"{log_w}x{log_h}+0+0")
        overlay.attributes('-topmost', True)
        overlay.attributes('-alpha', 0.25)
        overlay.configure(bg='#0F172A')
        overlay.update()

        canvas = tk.Canvas(overlay, cursor='crosshair', bg='#0F172A',
                           highlightthickness=0)
        canvas.pack(fill='both', expand=True)

        state = {'p1': None, 'marker': None, 'guide': None, 'region': None}

        guide_id = canvas.create_text(
            log_w // 2, 36,
            text="① 그리드 좌상단 모서리를 클릭하세요   |   ESC: 취소",
            fill='#FCD34D', font=(FONT, 14, 'bold'))

        hline = canvas.create_line(0, 0, log_w, 0, fill='#2563EB', width=1, dash=(4, 4))
        vline = canvas.create_line(0, 0, 0, log_h, fill='#2563EB', width=1, dash=(4, 4))

        def on_move(e):
            canvas.coords(hline, 0, e.y, log_w, e.y)
            canvas.coords(vline, e.x, 0, e.x, log_h)
            if state['p1']:
                x0, y0 = state['p1']
                if state['guide']:
                    canvas.delete(state['guide'])
                state['guide'] = canvas.create_rectangle(
                    x0, y0, e.x, e.y,
                    outline='#22C55E', width=2, fill='#ffffff', stipple='gray12')

        def on_click(e):
            if state['p1'] is None:
                state['p1'] = (e.x, e.y)
                state['marker'] = canvas.create_oval(
                    e.x - 5, e.y - 5, e.x + 5, e.y + 5,
                    fill='#22C55E', outline='#ffffff', width=2)
                canvas.itemconfig(guide_id,
                    text="② 그리드 우하단 모서리를 클릭하세요   |   ESC: 취소")
            else:
                x0, y0 = state['p1']
                x1, y1 = e.x, e.y
                lx, ly = min(x0, x1), min(y0, y1)
                rx, ry = max(x0, x1), max(y0, y1)
                state['region'] = (
                    int(lx * dpi_x), int(ly * dpi_y),
                    int((rx - lx) * dpi_x), int((ry - ly) * dpi_y),
                )
                overlay.destroy()

        canvas.bind('<Motion>',        on_move)
        canvas.bind('<ButtonPress-1>', on_click)
        overlay.bind('<Escape>', lambda _: overlay.destroy())
        canvas.focus_set()

        self.wait_window(overlay)
        self.deiconify()

        region = state['region']
        if region and region[2] > 10 and region[3] > 10:
            self.region = region
            self.region_lbl.config(
                text=f"({region[0]}, {region[1]})  {region[2]}×{region[3]}px",
                fg=C['success'])
            self.start_btn.config(state='normal')
            self._log(f"영역 선택: x={region[0]} y={region[1]} {region[2]}×{region[3]}")
            self.after(100, self._preview_region)
        else:
            self._log("영역 선택 취소")

    # ── 그리드 미리보기 ─────────────────────────────────────────────
    def _preview_region(self):
        if not self.region:
            self._log("영역을 먼저 선택하세요")
            return
        try:
            rows = self.rows_v.get(); cols = self.cols_v.get()
        except tk.TclError:
            return

        try:
            shot = pyautogui.screenshot(region=self.region)
        except Exception as e:
            self._log(f"미리보기 오류: {e}")
            return

        sw, sh = shot.size
        cell_w = sw / cols
        cell_h = sh / rows

        draw = ImageDraw.Draw(shot)
        for i in range(1, cols):
            x = int(i * cell_w)
            draw.line([(x, 0), (x, sh)], fill=(37, 99, 235), width=1)
        for i in range(1, rows):
            y = int(i * cell_h)
            draw.line([(0, y), (sw, y)], fill=(37, 99, 235), width=1)
        draw.rectangle([(0, 0), (sw - 1, sh - 1)], outline=(37, 99, 235), width=2)

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 500, 350
        scale = min(cw / sw, ch / sh)
        dw, dh = int(sw * scale), int(sh * scale)
        display = shot.resize((dw, dh), Image.LANCZOS)

        img_tk = ImageTk.PhotoImage(display)
        self.canvas._preview = img_tk
        self.canvas.delete('all')
        self._cell_ids = None
        self.canvas.create_image(cw // 2, ch // 2, image=img_tk, anchor='center')
        self.canvas.create_text(
            cw // 2, ch // 2 + dh // 2 + 14,
            text=f"미리보기  {cols}열 × {rows}행  |  셀 구분선이 게임 칸과 일치하는지 확인하세요",
            fill=C['secondary'], font=(FONT, 8))

        self._log(f"미리보기: {cols}×{rows} 그리드")

    # ═══════════════════════════════════════════════════════════════
    #  시작 / 정지
    # ═══════════════════════════════════════════════════════════════

    def _start(self):
        if not self.region: return
        try:
            rows = self.rows_v.get(); cols = self.cols_v.get(); mines = self.mines_v.get()
        except tk.TclError:
            self._log("설정값 오류"); return

        self.stop_evt.clear()
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.region_btn.config(state='disabled')
        self.learn_btn.config(state='disabled')
        self._log(f"시작: {rows}행×{cols}열, 지뢰{mines}개")

        self.worker = threading.Thread(
            target=self._game_loop,
            args=(rows, cols, mines),
            daemon=True)
        self.worker.start()

    def _stop(self):
        self.stop_evt.set()
        self._set_idle()
        self._log("정지")

    def _set_idle(self):
        self.start_btn.config(state='normal' if self.region else 'disabled')
        self.stop_btn.config(state='disabled')
        self.region_btn.config(state='normal')
        self.learn_btn.config(state='normal')

    # ── 게임 루프 ────────────────────────────────────────────────
    def _game_loop(self, rows, cols, mines):
        self._log("보드 분석 중...")
        try:
            detector = BoardDetector(self.region, rows, cols,
                                     references=self._saved_refs)
            solver = MinesweeperSolver(rows, cols, mines)
        except Exception as e:
            self._log(f"초기화 오류: {e}")
            self.after(0, self._set_idle)
            return

        mode = "레퍼런스" if self._saved_refs else "베이스라인"
        self._log(f"준비 완료 ({mode} 모드) — 풀이 시작")

        move_count   = 0
        stall_count  = 0
        prev_unknown = rows * cols

        while not self.stop_evt.is_set():
            try:
                board = detector.capture_board()
            except pyautogui.FailSafeException:
                self._log("FAILSAFE: 마우스를 화면 끝으로 이동해 중단됨")
                break
            except Exception as e:
                self._log(f"캡처 오류: {e}")
                break

            try: self.board_q.put_nowait(board)
            except queue.Full: pass

            unknown_cnt = sum(c == UNKNOWN for row in board for c in row)
            flag_cnt    = sum(c == FLAG    for row in board for c in row)

            self.after(0, self.status_lbl.config, {
                'text': f"이동: {move_count}  |  미확인: {unknown_cnt}  |  깃발: {flag_cnt}/{mines}"
            })

            if any(c == MINE for row in board for c in row):
                self._log(f"지뢰 밟음! ({move_count}번 이동)")
                self.after(0, self._set_idle); break

            if unknown_cnt + flag_cnt == mines:
                self._log(f"클리어! ({move_count}번 이동)")
                self.after(0, self._set_idle); break

            solver.update(board)
            safe_cells, mine_cells = solver.solve()

            if not safe_cells and not mine_cells:
                guess = solver.best_guess()
                if guess is None:
                    self._log("더 이상 풀 수 없습니다.")
                    self.after(0, self._set_idle); break
                self._log(f"[추측] ({guess[0]},{guess[1]})")
                safe_cells = [guess]
            else:
                parts = []
                if mine_cells: parts.append(f"지뢰 {len(mine_cells)}개")
                if safe_cells: parts.append(f"안전 {len(safe_cells)}개")
                if parts: self._log(f"[확정] {', '.join(parts)}")

            delay = self.speed_v.get() / 1000

            try:
                for r, c in mine_cells:
                    if self.stop_evt.is_set(): break
                    if board[r][c] == UNKNOWN:
                        x, y = detector.cell_center(r, c)
                        pyautogui.rightClick(x, y)
                        move_count += 1
                        time.sleep(delay * 0.35)

                for r, c in safe_cells:
                    if self.stop_evt.is_set(): break
                    if board[r][c] == UNKNOWN:
                        x, y = detector.cell_center(r, c)
                        pyautogui.click(x, y)
                        move_count += 1
                        time.sleep(delay)

            except pyautogui.FailSafeException:
                self._log("FAILSAFE 중단")
                self.after(0, self._set_idle); break

            time.sleep(max(0.08, delay * 0.3))

            try:
                new_board = detector.capture_board()
                new_unknown = sum(c == UNKNOWN for row in new_board for c in row)
                if new_unknown >= prev_unknown and not mine_cells:
                    stall_count += 1
                    if stall_count >= 4:
                        self._log("보드가 변하지 않습니다. 영역·설정을 확인하세요.")
                        self.after(0, self._set_idle); break
                else:
                    stall_count = 0
                prev_unknown = new_unknown
            except Exception:
                pass

        self.after(0, self._set_idle)

    # ═══════════════════════════════════════════════════════════════
    #  캔버스 그리기
    # ═══════════════════════════════════════════════════════════════

    def _redraw(self):
        self._cell_ids = None
        if self._learning:
            self._show_learn_canvas()
        elif self._board_cache:
            self._draw_board(self._board_cache)

    def _init_cells(self, rows, cols):
        self.canvas.delete('all')
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        cw_cell = cw / cols
        ch_cell = ch / rows
        fs = max(6, int(min(cw_cell, ch_cell) * 0.52))

        ids = []
        for r in range(rows):
            row_ids = []
            for c in range(cols):
                x1, y1 = c * cw_cell, r * ch_cell
                x2, y2 = x1 + cw_cell, y1 + ch_cell
                rid = self.canvas.create_rectangle(x1, y1, x2, y2,
                          fill='#CBD5E1', outline='#E2E8F0', width=1)
                tid = self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2,
                          text='', fill='#1F2937',
                          font=(FONT, fs, 'bold'))
                row_ids.append((rid, tid, None))
            ids.append(row_ids)
        self._cell_ids = ids
        self._cell_grid = (rows, cols)

    def _draw_board(self, board):
        self._board_cache = board
        rows = len(board)
        if not rows: return
        cols = len(board[0])
        if not cols: return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10: return

        if self._cell_ids is None or self._cell_grid != (rows, cols):
            self._init_cells(rows, cols)

        for r in range(rows):
            for c in range(cols):
                state = board[r][c]
                rid, tid, last = self._cell_ids[r][c]
                if state == last:
                    continue

                if state in CELL_FILL:
                    fill = CELL_FILL[state]
                    text = {FLAG: 'F', MINE: 'X'}.get(state, '')
                    tcolor = '#1F2937' if state == UNKNOWN or state == EMPTY else '#FFFFFF'
                else:
                    fill   = '#FFFFFF'
                    text   = str(state)
                    tcolor = NUM_COLOR.get(state, '#1F2937')

                self.canvas.itemconfig(rid, fill=fill)
                self.canvas.itemconfig(tid, text=text, fill=tcolor)
                self._cell_ids[r][c] = (rid, tid, state)

    # ── 큐 폴링 ──────────────────────────────────────────────────
    def _poll(self):
        try:
            while True:
                board = self.board_q.get_nowait()
                if not self._learning:
                    self._draw_board(board)
        except queue.Empty:
            pass

        try:
            while True:
                msg = self.log_q.get_nowait()
                self.log_txt.config(state='normal')
                self.log_txt.insert('end', msg + '\n')
                self.log_txt.see('end')
                self.log_txt.config(state='disabled')
        except queue.Empty:
            pass

        self.after(50, self._poll)

    def _log(self, msg):
        self.log_q.put(msg)


# ════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    App().mainloop()
