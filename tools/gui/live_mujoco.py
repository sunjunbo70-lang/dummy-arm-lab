"""MuJoCo visual upper controller. Starts read-only; enable following in UI."""
from pathlib import Path
import argparse
import json
import sys
import time
import tkinter as tk
from tkinter import ttk,messagebox
import numpy as np
import mujoco
from PIL import Image,ImageTk

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dummy_loop.live_control import LiveController,LOWER,UPPER,HOME
from dummy_loop.sim_backend import SimRobot, V2_MODEL


MIN_RENDER=(240,180)
MAX_RENDER=(1920,1440)


def enable_high_dpi():
    """Windows：声明进程感知 DPI，避免系统把整个窗口按位图放大（发糊）或按 100% 画（字太小）。"""
    if sys.platform!='win32': return
    import ctypes
    try: ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try: ctypes.windll.user32.SetProcessDPIAware()
        except Exception: pass


def close_renderer(renderer):
    """在渲染器自己的 GL 上下文里先释放 MjrContext，再销毁上下文。"""
    gl=getattr(renderer,'_gl_context',None); mjr=getattr(renderer,'_mjr_context',None)
    if gl is None or mjr is None:
        renderer.close(); return
    gl.make_current(); mjr.free(); renderer._mjr_context=None
    gl.free(); renderer._gl_context=None


class Scene:
    def __init__(self):
        # Dummy V2 模型：运动学与固件 V2 DH 完全一致，模型零位 = 固件 HOME，正方向与固件相同，
        # 所以固件角度直接换算即可显示，不需要任何符号表（tests/test_dummy_v2_model.py 核对）。
        self.model=SimRobot(V2_MODEL).model
        self.sign=np.ones(6); self.fallback=False
        self.data=mujoco.MjData(self.model)
        # Display only: forward kinematics, no dynamics or simulated force output.
        self.model.jnt_limited[:]=0
        self.ids=[self.model.jnt_qposadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,f'Joint{i}')] for i in range(1,7)]
        # 离屏缓冲上限：窗口放大 / 全屏时画面跟着变大，不再固定 540×430。
        self.model.vis.global_.offwidth=MAX_RENDER[0]; self.model.vis.global_.offheight=MAX_RENDER[1]
        self.size=(540,430)
        self.renderer=mujoco.Renderer(self.model,height=self.size[1],width=self.size[0])
        self.camera=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(self.camera)
        self.camera.lookat[:]=[.06,0,.19]; self.camera.distance=.85
        self.camera.azimuth=-135; self.camera.elevation=-18

    def resize(self,width,height):
        """按显示区域重建渲染器（尺寸夹在 [MIN_RENDER, MAX_RENDER] 内）。返回是否真的改了尺寸。"""
        w=int(np.clip(width,MIN_RENDER[0],MAX_RENDER[0])); h=int(np.clip(height,MIN_RENDER[1],MAX_RENDER[1]))
        if (w,h)==self.size: return False
        # 先释放旧渲染器、再建新的。反过来会让旧 MjrContext 在新 GL 上下文里删对象，
        # 把新渲染器的帧缓冲删掉，画面变黑。
        close_renderer(self.renderer)
        self.renderer=mujoco.Renderer(self.model,height=h,width=w); self.size=(w,h)
        return True

    def frame(self,firmware_deg):
        # V2 模型：q = deg2rad(固件角 - HOME)。仅用于显示，从不用于生成硬件目标。
        self.data.qpos[self.ids]=self.sign*np.deg2rad(np.asarray(firmware_deg)-HOME)
        mujoco.mj_forward(self.model,self.data)
        self.renderer.update_scene(self.data,camera=self.camera)
        return self.renderer.render().copy()


class App:
    def __init__(self,root,auto_connect=True):
        self.root=root; root.title('Dummy · MuJoCo 实机同步上位机（V2 模型）')
        self.setup_scaling()
        sw,sh=root.winfo_screenwidth(),root.winfo_screenheight()
        w,h=min(int(sw*.8),max(1150,int(sw*.6))),min(int(sh*.85),max(880,int(sh*.7)))
        root.geometry(f'{w}x{h}+{(sw-w)//2}+{max(0,(sh-h)//3)}'); root.minsize(900,680)
        self.control=LiveController(ROOT/f'outputs/live_serial_{time.time_ns()}.jsonl')
        self.scene=Scene()
        self.generation=-1; self.setting=False; self.closing=False
        self.values=[tk.DoubleVar(value=v) for v in HOME]; self.readouts=[]; self.target_readouts=[]
        self.images=[]; self.last_q=HOME.copy(); self.last_draw=0.; self.drag=None; self.pending_size=None
        style=ttk.Style(); style.configure('TButton',padding=(10,5))
        style.configure('Stop.TButton',foreground='white',background='#af3038',padding=(16,5))
        style.map('Stop.TButton',background=[('active','#c8414a')])
        toolbar=ttk.Frame(root,padding=(8,8,8,4)); toolbar.pack(side='top',fill='x')
        ttk.Button(toolbar,text='连接 / 重新连接（只读）',command=self.control.connect).pack(side='left')
        self.arm_button=ttk.Button(toolbar,text='开启实时跟随',command=self.arm); self.arm_button.pack(side='left',padx=8)
        ttk.Button(toolbar,text='停止跟随  [空格]',style='Stop.TButton',command=self.stop).pack(side='left')
        ttk.Button(toolbar,text='目标回到当前实机',command=self.reset_target).pack(side='left',padx=8)
        self.home_button=ttk.Button(toolbar,text='直立复位',command=lambda:self.go_pose('home')); self.home_button.pack(side='left',padx=3)
        self.fold_button=ttk.Button(toolbar,text='折叠',command=lambda:self.go_pose('fold')); self.fold_button.pack(side='left',padx=3)
        ttk.Button(toolbar,text='断开',command=self.control.close).pack(side='right')
        options=ttk.Frame(root,padding=(8,2)); options.pack(side='top',fill='x')
        self.speed=tk.DoubleVar(value=5); self.speed_text=tk.StringVar(value='跟随速度：5.0°/秒')
        ttk.Label(options,textvariable=self.speed_text,width=16).pack(side='left')
        ttk.Scale(options,from_=1,to=20,variable=self.speed,command=self.change_speed).pack(side='left',fill='x',expand=True,padx=(4,12))
        ttk.Label(options,text='上位机目标速率；实际速度受反馈与控制器限制').pack(side='left')
        auto=ttk.Frame(root,padding=(8,2)); auto.pack(side='top',fill='x')
        self.auto_button=ttk.Button(auto,text='启动六轴全范围线性往复',command=self.start_auto)
        self.auto_button.pack(side='left',padx=2)
        self.auto_result=tk.StringVar(value='各轴覆盖软件上下限；持续运行，按停止结束')
        ttk.Label(auto,textvariable=self.auto_result).pack(side='left',padx=6)
        self.status=tk.StringVar(value='初始化 MuJoCo…')
        ttk.Label(root,textvariable=self.status,style='Status.TLabel',padding=(10,6)).pack(side='top',fill='x')
        # 底部先 pack（side=bottom），中间的三维视图最后 pack 并 expand，占满剩余空间。
        notes=ttk.Frame(root,padding=(12,2,12,8)); notes.pack(side='bottom',fill='x')
        n1=ttk.Label(notes,text='拖动图像旋转视角，滚轮缩放；Ctrl+滚轮 或 Ctrl+= / Ctrl+- 调整界面字号。'
                  '开启跟随后，拖动滑块即发往实体；自动测试期间锁定手动滑块。'); n1.pack(anchor='w',fill='x')
        n2=ttk.Label(notes,text='画面为 V2 模型：运动学按 V2 固件 DH，与固件角度一一对应；外形与质量来自 CAD，未经实机核对，'
                  '不用于碰撞判断。显示的是固件缓存角度，非带传感器时间戳的反馈。软件停止不能替代实体停止。',
                  foreground='#8b5318'); n2.pack(anchor='w',fill='x')
        notes.bind('<Configure>',lambda e:[n.configure(wraplength=max(200,e.width-24)) for n in (n1,n2)])
        table=ttk.Frame(root,padding=(8,4)); table.pack(side='bottom',fill='x')
        table.columnconfigure(3,weight=1)
        for col,title in enumerate(('关节','反馈 °','目标 °','拖动调整目标（角度制）')):
            ttk.Label(table,text=title).grid(row=0,column=col,sticky='w',padx=8)
        self.sliders=[]
        for i in range(6):
            ttk.Label(table,text=f'J{i+1}',width=4).grid(row=i+1,column=0,padx=8)
            read=ttk.Label(table,text='—',width=9); read.grid(row=i+1,column=1,padx=8); self.readouts.append(read)
            out=ttk.Label(table,text=f'{HOME[i]:.2f}',width=9); out.grid(row=i+1,column=2,padx=8); self.target_readouts.append(out)
            slider=ttk.Scale(table,from_=LOWER[i],to=UPPER[i],variable=self.values[i],command=self.changed)
            slider.grid(row=i+1,column=3,sticky='ew',pady=3,padx=(0,8)); self.sliders.append(slider)
        frames=ttk.Frame(root,padding=(8,0)); frames.pack(side='top',fill='both',expand=True)
        frames.rowconfigure(0,weight=1); frames.columnconfigure((0,1),weight=1,uniform='view')
        self.views=[]; self.view_boxes=[]
        for col,title in enumerate(('实机反馈姿态 · 原生 USB 角度','目标预览 · 滑块设定角度')):
            f=ttk.LabelFrame(frames,text=title,padding=2); f.grid(row=0,column=col,sticky='nsew',padx=4,pady=2)
            box=tk.Frame(f,bg='black'); box.pack(fill='both',expand=True); box.pack_propagate(False)
            label=tk.Label(box,bg='black',bd=0); label.place(relx=.5,rely=.5,anchor='center')
            self.views.append(label); self.view_boxes.append(box)
            for w in (box,label):
                w.bind('<ButtonPress-1>',self.drag_start); w.bind('<B1-Motion>',self.drag_move)
                w.bind('<MouseWheel>',self.zoom)
                w.bind('<Button-4>',lambda e:self.zoom(e,1)); w.bind('<Button-5>',lambda e:self.zoom(e,-1))
            box.bind('<Configure>',self.view_resized)
        root.bind_all('<Control-MouseWheel>',lambda e:self.font_step(1 if e.delta>0 else -1))
        root.bind_all('<Control-equal>',lambda e:self.font_step(1)); root.bind_all('<Control-plus>',lambda e:self.font_step(1))
        root.bind_all('<Control-minus>',lambda e:self.font_step(-1))
        root.bind('<space>',lambda event:self.stop()); root.bind('<Escape>',lambda event:self.stop())
        root.protocol('WM_DELETE_WINDOW',self.close)
        if auto_connect: self.control.connect()
        self.tick()

    def setup_scaling(self):
        """按屏幕 DPI 设置 Tk 缩放和字号：高分屏、全屏时字和控件跟着变大。"""
        import tkinter.font as tkfont
        ttk.Style().theme_use('clam')      # 先定主题，后面的样式设置才不会被切换主题冲掉
        dpi=self.root.winfo_fpixels('1i')
        self.root.tk.call('tk','scaling',dpi/72.0)
        # 基础字号（磅）：按屏幕物理高度适度放大，4K / 2K 屏上不至于太小
        sh=self.root.winfo_screenheight()/dpi*96        # 换算成 96 DPI 下的逻辑像素高度
        self.font_size=10 if sh<=1100 else 11 if sh<=1500 else 12
        self.fonts=[tkfont.nametofont(n) for n in ('TkDefaultFont','TkTextFont','TkMenuFont','TkHeadingFont','TkCaptionFont','TkSmallCaptionFont','TkIconFont','TkTooltipFont')]
        family='Microsoft YaHei UI' if sys.platform=='win32' else None
        for f in self.fonts:
            if family: f.configure(family=family)
        self.status_font=tkfont.Font(family=family or self.fonts[0].actual('family'),size=self.font_size+1)
        ttk.Style().configure('Status.TLabel',font=self.status_font)
        self.apply_font_size()

    def apply_font_size(self):
        for f in self.fonts: f.configure(size=self.font_size)
        self.status_font.configure(size=self.font_size+1)

    def font_step(self,step):
        self.font_size=int(np.clip(self.font_size+step,8,20)); self.apply_font_size()

    def view_resized(self,event):
        # 两个视图等宽等高，取第一个的尺寸；拖动窗口时连续触发，稍等尺寸稳定再重建渲染器
        box=self.view_boxes[0]
        self.pending_size=(box.winfo_width()-4,box.winfo_height()-4)
        if getattr(self,'_resize_job',None): self.root.after_cancel(self._resize_job)
        self._resize_job=self.root.after(150,self.apply_view_size)

    def apply_view_size(self):
        self._resize_job=None
        if self.pending_size and self.pending_size[0]>10 and self.pending_size[1]>10:
            self.scene.resize(*self.pending_size)

    def set_values(self,q):
        self.setting=True
        for var,v in zip(self.values,q): var.set(float(v))
        self.setting=False

    def changed(self,*args):
        if self.setting or self.closing: return
        target=np.array([v.get() for v in self.values])
        try:
            if self.control.snapshot()['connected']: self.control.set_target(target)
        except Exception as exc: self.status.set(str(exc))

    def arm(self):
        try: self.control.arm()
        except Exception as exc: messagebox.showerror('不能开启跟随',str(exc))

    def change_speed(self,*args):
        value=round(self.speed.get(),1); self.control.set_speed(value)
        self.speed_text.set(f'跟随速度：{value:.1f}°/秒')

    def start_auto(self):
        try: self.control.start_auto()
        except Exception as exc: messagebox.showerror('不能启动自动测试',str(exc))

    def go_pose(self,kind):
        try: self.control.request_pose(kind)
        except Exception as exc: messagebox.showerror('不能切换姿态',str(exc))

    def stop(self): self.control.stop()
    def reset_target(self):
        state=self.control.snapshot()
        if state['q'] is not None:
            q=np.clip(state['q'],LOWER,UPPER); self.set_values(q); self.control.set_target(q)

    def drag_start(self,event): self.drag=(event.x,event.y)
    def drag_move(self,event):
        if self.drag:
            self.scene.camera.azimuth-=(event.x-self.drag[0])*.4
            self.scene.camera.elevation=np.clip(self.scene.camera.elevation-(event.y-self.drag[1])*.3,-85,85)
            self.drag=(event.x,event.y)
    def zoom(self,event,direction=None):
        up=(event.delta>0) if direction is None else direction>0
        self.scene.camera.distance=np.clip(self.scene.camera.distance*(.9 if up else 1.1),.3,2.)

    def tick(self):
        if self.closing: return
        self.control.heartbeat(); state=self.control.snapshot()
        if state['q'] is not None: self.last_q=state['q']
        if state['generation']!=self.generation and state['target'] is not None:
            self.generation=state['generation']; self.set_values(state['target'])
        age=time.monotonic()-state['rx_at']
        stale=state['connected'] and age>.75
        if stale:
            self.control.expire_feedback()
            state=self.control.snapshot()
        self.status.set(state['status']+(f'  |  最近反馈响应 {age:.2f}s 前' if state['rx_at'] else ''))
        self.arm_button.configure(state='normal' if state['connected'] and not state['active'] else 'disabled')
        for button in (self.home_button,self.fold_button): button.configure(state='normal' if state['connected'] and not stale else 'disabled')
        self.auto_button.configure(state='normal' if state['connected'] and not state['active'] and not state['automatic'] else 'disabled')
        self.auto_result.set(state['auto_progress'] or '各轴覆盖软件上下限；持续运行，按停止结束')
        for slider in self.sliders: slider.configure(state='disabled' if state['automatic'] else 'normal')
        target=np.array([v.get() for v in self.values])
        for i in range(6):
            self.readouts[i].configure(text=f'{self.last_q[i]:.2f}' if state['q'] is not None else '—')
            self.target_readouts[i].configure(text=f'{target[i]:.2f}')
        try:
            images=[ImageTk.PhotoImage(Image.fromarray(self.scene.frame(q))) for q in (self.last_q,target)]
            for label,img in zip(self.views,images): label.configure(image=img)
            self.images=images
        except Exception:
            self.control.close(); raise
        self.root.after(80,self.tick)

    def close(self):
        self.closing=True; self.control.close(); self.status.set('正在停止并释放连接…')
        self.wait_closed()

    def wait_closed(self):
        if self.control.thread and self.control.thread.is_alive():
            self.root.after(100,self.wait_closed); return
        close_renderer(self.scene.renderer); self.root.destroy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--fullscreen',action='store_true',help='启动时最大化窗口')
    parser.add_argument('--smoke-test',action='store_true',help='Render home/fold poses to PNG without opening hardware')
    args=parser.parse_args()
    if args.smoke_test:
        scene=Scene()
        for name,q in [('home',HOME),('fold',[0,-75,180,0,0,0])]:
            Image.fromarray(scene.frame(q)).save(ROOT/f'outputs/live_{name}_preview.png')
        scene.renderer.close(); print('RENDER_OK'); return
    enable_high_dpi()
    root=tk.Tk(); App(root,auto_connect=not args.offline)
    if args.fullscreen: root.state('zoomed') if sys.platform=='win32' else root.attributes('-zoomed',True)
    root.mainloop()


if __name__=='__main__': main()
