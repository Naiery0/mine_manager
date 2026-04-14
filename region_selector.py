"""
화면에서 게임 보드 영역을 드래그로 선택하는 GUI.
스크린샷을 찍어서 tkinter 창에 보여주고, 사용자가 드래그해서 영역을 지정.
"""

import tkinter as tk
from PIL import Image, ImageTk
import pyautogui


class RegionSelector:
    def select(self):
        """스크린샷 위에서 드래그로 영역 선택. (x, y, w, h) 반환."""
        screenshot = pyautogui.screenshot()
        sw, sh = screenshot.size

        # 화면에 맞게 축소
        scale = min(1.0, 1400 / sw, 900 / sh)
        dw = int(sw * scale)
        dh = int(sh * scale)
        display = screenshot.resize((dw, dh), Image.LANCZOS)

        root = tk.Tk()
        root.title("게임 보드를 드래그해서 선택하세요  (ESC: 취소)")
        root.resizable(False, False)

        canvas = tk.Canvas(root, width=dw, height=dh, cursor="crosshair",
                           highlightthickness=0)
        canvas.pack()

        label = tk.Label(
            root,
            text="마우스를 드래그해서 지뢰찾기 보드 영역을 선택하세요",
            bg="#222", fg="white", pady=4
        )
        label.pack(fill="x")

        img_tk = ImageTk.PhotoImage(display)
        canvas.create_image(0, 0, image=img_tk, anchor="nw")

        state = {"start": None, "rect": None, "region": None}

        def on_press(e):
            state["start"] = (e.x, e.y)
            if state["rect"]:
                canvas.delete(state["rect"])
                state["rect"] = None

        def on_drag(e):
            if state["start"]:
                if state["rect"]:
                    canvas.delete(state["rect"])
                x0, y0 = state["start"]
                state["rect"] = canvas.create_rectangle(
                    x0, y0, e.x, e.y, outline="red", width=2
                )

        def on_release(e):
            if state["start"]:
                x0, y0 = state["start"]
                x1, y1 = e.x, e.y
                # 실제 화면 좌표로 변환
                rx = int(min(x0, x1) / scale)
                ry = int(min(y0, y1) / scale)
                rw = int(abs(x1 - x0) / scale)
                rh = int(abs(y1 - y0) / scale)
                state["region"] = (rx, ry, rw, rh)
                root.destroy()

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        root.bind("<Escape>", lambda e: root.destroy())

        root.mainloop()
        return state["region"]
