import asyncio
import json
import queue
import threading
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import messagebox, scrolledtext, ttk
from tkinter import font as tkfont
from typing import Any

import websockets


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8888


@dataclass
class ServerState:
    clients: set[Any] = field(default_factory=set)
    last_client: Any | None = None


class WebsocketTestServer:
    def __init__(self, host: str, port: int, gui_queue: queue.Queue[tuple[str, str]]):
        self.host = host
        self.port = port
        self.gui_queue = gui_queue
        self.state = ServerState()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.stop_event: asyncio.Event | None = None
        self.thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.loop is not None and self.stop_event is not None:
            self.loop.call_soon_threadsafe(self.stop_event.set)

    def send(self, payload: dict[str, Any]) -> None:
        if self.loop is None or not self.is_running:
            self._emit("status", "server is not running")
            return

        future = asyncio.run_coroutine_threadsafe(self._send(payload), self.loop)
        future.add_done_callback(self._send_done)

    def _thread_main(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._run())
        except Exception as e:
            self._emit("status", f"server error: {e}")
        finally:
            self.loop.close()
            self.loop = None
            self.stop_event = None
            self._emit("status", "server stopped")

    async def _run(self) -> None:
        self.stop_event = asyncio.Event()
        async with websockets.serve(self._handler, self.host, self.port):
            self._emit("status", f"listening on ws://{self.host}:{self.port}")
            await self.stop_event.wait()

    async def _handler(self, ws: Any) -> None:
        peer = ws.remote_address
        self.state.clients.add(ws)
        self.state.last_client = ws
        self._emit("status", f"client connected: {peer}")

        try:
            async for message in ws:
                self._emit("received", self._pretty_message(message))
        finally:
            self.state.clients.discard(ws)
            if self.state.last_client is ws:
                self.state.last_client = next(iter(self.state.clients), None)
            self._emit("status", f"client disconnected: {peer}")

    async def _send(self, payload: dict[str, Any]) -> None:
        if self.state.last_client is None:
            self._emit("status", "no websocket client is connected")
            return

        await self.state.last_client.send(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
        self._emit("sent", json.dumps(payload, indent=2, ensure_ascii=False))

    def _send_done(self, future: asyncio.Future) -> None:
        try:
            future.result()
        except Exception as e:
            self._emit("status", f"send failed: {e}")

    def _pretty_message(self, message: str) -> str:
        try:
            return json.dumps(json.loads(message), indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            return message

    def _emit(self, event_type: str, text: str) -> None:
        self.gui_queue.put((event_type, text))


class WebsocketGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gym Vision WebSocket Test Server")
        self.geometry("1240x800")
        self.minsize(1040, 680)
        self.configure(bg="#f4f6f8")
        self._configure_style()

        self.gui_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.server: WebsocketTestServer | None = None

        self.host_var = tk.StringVar(value=DEFAULT_HOST)
        self.port_var = tk.IntVar(value=DEFAULT_PORT)
        self.member_id_var = tk.StringVar(value="1008")
        self.cam_ip_var = tk.StringVar(value="192.168.1.64")
        self.address_var = tk.StringVar(value="/tmp/gym_vision_backup")
        self.status_var = tk.StringVar(value="stopped")

        self._build_ui()
        self.after(100, self._poll_gui_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        self.default_font = tkfont.Font(family="DejaVu Sans", size=12)
        self.heading_font = tkfont.Font(family="DejaVu Sans", size=18, weight="bold")
        self.section_font = tkfont.Font(family="DejaVu Sans", size=12, weight="bold")
        self.button_font = tkfont.Font(family="DejaVu Sans", size=11, weight="bold")
        self.text_font = tkfont.Font(family="DejaVu Sans Mono", size=11)

        self.option_add("*Font", self.default_font)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background="#f4f6f8")
        style.configure("Panel.TLabelframe", background="#ffffff", borderwidth=1, relief="solid")
        style.configure(
            "Panel.TLabelframe.Label",
            background="#f4f6f8",
            foreground="#263238",
            font=self.section_font,
            padding=(4, 2),
        )
        style.configure("TLabel", background="#ffffff", foreground="#263238", font=self.default_font)
        style.configure("Muted.TLabel", background="#ffffff", foreground="#607d8b", font=self.default_font)
        style.configure("Header.TLabel", background="#f4f6f8", foreground="#102027", font=self.heading_font)
        style.configure("Status.TLabel", background="#e8f5e9", foreground="#1b5e20", padding=(12, 6), font=self.section_font)
        style.configure("TEntry", padding=(7, 5), font=self.default_font)
        style.configure("TButton", padding=(12, 7), font=self.button_font)
        style.configure("Primary.TButton", padding=(14, 8), font=self.button_font)
        style.configure("Danger.TButton", padding=(12, 7), font=self.button_font)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=14, style="App.TFrame")
        root.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(root, style="App.TFrame")
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header, text="Gym Vision WebSocket Test Server", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, textvariable=self.status_var, style="Status.TLabel").pack(side=tk.RIGHT)

        server_frame = ttk.LabelFrame(root, text="Server", padding=12, style="Panel.TLabelframe")
        server_frame.pack(fill=tk.X)

        ttk.Label(server_frame, text="Host").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(server_frame, textvariable=self.host_var, width=20).grid(row=0, column=1, padx=(8, 18), ipady=2)
        ttk.Label(server_frame, text="Port").grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(server_frame, textvariable=self.port_var, width=9).grid(row=0, column=3, padx=(8, 18), ipady=2)
        ttk.Button(server_frame, text="Start Server", command=self._start_server, style="Primary.TButton").grid(row=0, column=4, padx=(0, 8))
        ttk.Button(server_frame, text="Stop", command=self._stop_server).grid(row=0, column=5, padx=(0, 8))
        ttk.Label(server_frame, text="Start this before running main.py.", style="Muted.TLabel").grid(row=0, column=6, padx=(12, 0), sticky=tk.W)
        server_frame.columnconfigure(6, weight=1)

        command_frame = ttk.LabelFrame(root, text="Command Builder", padding=12, style="Panel.TLabelframe")
        command_frame.pack(fill=tk.X, pady=(12, 0))

        ttk.Label(command_frame, text="memberID").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(command_frame, textvariable=self.member_id_var, width=14).grid(row=0, column=1, padx=(8, 18), sticky=tk.W, ipady=2)
        ttk.Label(command_frame, text="camIP").grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(command_frame, textvariable=self.cam_ip_var, width=22).grid(row=0, column=3, padx=(8, 18), sticky=tk.W, ipady=2)
        ttk.Label(command_frame, text="Address").grid(row=0, column=4, sticky=tk.W)
        ttk.Entry(command_frame, textvariable=self.address_var, width=36).grid(row=0, column=5, padx=(8, 0), sticky=tk.EW, ipady=2)
        command_frame.columnconfigure(5, weight=1)

        button_groups = ttk.Frame(command_frame)
        button_groups.grid(row=1, column=0, columnspan=6, sticky=tk.EW, pady=(12, 0))
        for col in range(4):
            button_groups.columnconfigure(col, weight=1, uniform="commands")

        self._add_button_group(
            button_groups,
            0,
            "System",
            [
                ("Connection", lambda: self._send({"Type": "connection"})),
                ("Count DB", lambda: self._send({"Type": "countDB"})),
                ("Get List", lambda: self._send({"Type": "getList"})),
            ],
        )
        self._add_button_group(
            button_groups,
            1,
            "Camera",
            [
                ("Check All", self._send_check_all),
                ("Check IP", self._send_check_ip),
            ],
        )
        self._add_button_group(
            button_groups,
            2,
            "Member",
            [
                ("Register", self._send_reg),
                ("Delete", self._send_del),
                ("Delete All", lambda: self._send({"Type": "delAll"})),
            ],
        )
        self._add_button_group(
            button_groups,
            3,
            "Database",
            [
                ("Get DB", self._send_get_db),
                ("Restore DB", self._send_restore_db),
            ],
        )

        raw_frame = ttk.LabelFrame(root, text="Raw JSON", padding=12, style="Panel.TLabelframe")
        raw_frame.pack(fill=tk.X, pady=(12, 0))
        raw_frame.columnconfigure(0, weight=1)
        self.raw_text = tk.Text(
            raw_frame,
            height=4,
            wrap=tk.WORD,
            font=self.text_font,
            bg="#fbfcfd",
            fg="#102027",
            insertbackground="#102027",
            relief=tk.FLAT,
            padx=10,
            pady=8,
        )
        self.raw_text.grid(row=0, column=0, sticky=tk.EW)
        self.raw_text.insert("1.0", '{"Type":"connection"}')
        ttk.Button(raw_frame, text="Send Raw JSON", command=self._send_raw, style="Primary.TButton").grid(row=0, column=1, padx=(10, 0), sticky=tk.NS)

        display_frame = ttk.Frame(root, style="App.TFrame")
        display_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.rowconfigure(0, weight=1)

        sent_frame = ttk.LabelFrame(display_frame, text="Sent To Vision Service", padding=8, style="Panel.TLabelframe")
        sent_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 6))
        received_frame = ttk.LabelFrame(display_frame, text="Received From Vision Service", padding=8, style="Panel.TLabelframe")
        received_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=(6, 0))

        self.sent_display = scrolledtext.ScrolledText(
            sent_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=self.text_font,
            bg="#f8fbff",
            fg="#0d2538",
            relief=tk.FLAT,
            padx=10,
            pady=10,
        )
        self.sent_display.pack(fill=tk.BOTH, expand=True)
        self.received_display = scrolledtext.ScrolledText(
            received_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=self.text_font,
            bg="#fbfff8",
            fg="#1b2d10",
            relief=tk.FLAT,
            padx=10,
            pady=10,
        )
        self.received_display.pack(fill=tk.BOTH, expand=True)

        bottom = ttk.Frame(root, style="App.TFrame")
        bottom.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(bottom, text="Clear Sent", command=lambda: self._clear(self.sent_display)).pack(side=tk.LEFT)
        ttk.Button(bottom, text="Clear Received", command=lambda: self._clear(self.received_display)).pack(side=tk.LEFT, padx=(6, 0))

    def _add_button_group(self, parent: ttk.Frame, column: int, title: str, buttons: list[tuple[str, Any]]) -> None:
        frame = ttk.LabelFrame(parent, text=title, padding=8, style="Panel.TLabelframe")
        frame.grid(row=0, column=column, sticky=tk.NSEW, padx=(0 if column == 0 else 8, 0))
        frame.columnconfigure(0, weight=1)

        for row, (label, command) in enumerate(buttons):
            ttk.Button(frame, text=label, command=command).grid(row=row, column=0, sticky=tk.EW, pady=(0, 6))

    def _start_server(self) -> None:
        if self.server is not None and self.server.is_running:
            return

        try:
            port = int(self.port_var.get())
        except (TypeError, ValueError):
            messagebox.showerror("Invalid port", "Port must be an integer.")
            return

        self.server = WebsocketTestServer(self.host_var.get().strip(), port, self.gui_queue)
        self.server.start()

    def _stop_server(self) -> None:
        if self.server is not None:
            self.server.stop()

    def _send(self, payload: dict[str, Any]) -> None:
        if self.server is None:
            self._append(self.sent_display, "server is not running")
            return
        self.server.send(payload)

    def _send_check_all(self) -> None:
        self._send({"Type": "checkCam"})

    def _send_check_ip(self) -> None:
        cam_ip = self.cam_ip_var.get().strip()
        self._send({"Type": "checkCam", **({"camIP": cam_ip} if cam_ip else {})})

    def _send_reg(self) -> None:
        try:
            member_id = int(self.member_id_var.get())
        except ValueError:
            messagebox.showerror("Invalid memberID", "memberID must be an integer.")
            return

        payload: dict[str, Any] = {"Type": "reg", "memberID": member_id}
        cam_ip = self.cam_ip_var.get().strip()
        if cam_ip:
            payload["camIP"] = cam_ip
        self._send(payload)

    def _send_del(self) -> None:
        try:
            member_id = int(self.member_id_var.get())
        except ValueError:
            messagebox.showerror("Invalid memberID", "memberID must be an integer.")
            return
        self._send({"Type": "del", "memberID": member_id})

    def _send_get_db(self) -> None:
        self._send({"Type": "getDB", "Address": self.address_var.get().strip()})

    def _send_restore_db(self) -> None:
        self._send({"Type": "restoreDB", "Address": self.address_var.get().strip()})

    def _send_raw(self) -> None:
        raw = self.raw_text.get("1.0", tk.END).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            messagebox.showerror("Invalid JSON", str(e))
            return
        if not isinstance(payload, dict):
            messagebox.showerror("Invalid JSON", "Top-level JSON value must be an object.")
            return
        self._send(payload)

    def _poll_gui_queue(self) -> None:
        while True:
            try:
                event_type, text = self.gui_queue.get_nowait()
            except queue.Empty:
                break

            if event_type == "sent":
                self._append(self.sent_display, text)
            elif event_type == "received":
                self._append(self.received_display, text)
            else:
                self.status_var.set(text)

        self.after(100, self._poll_gui_queue)

    def _append(self, widget: scrolledtext.ScrolledText, text: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.insert(tk.END, f"{text}\n\n")
        widget.see(tk.END)
        widget.configure(state=tk.DISABLED)

    def _clear(self, widget: scrolledtext.ScrolledText) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.configure(state=tk.DISABLED)

    def _on_close(self) -> None:
        self._stop_server()
        self.after(150, self.destroy)


if __name__ == "__main__":
    WebsocketGui().mainloop()
