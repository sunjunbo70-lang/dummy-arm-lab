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
from dummy_loop.sim_backend import SimRobot, studio_meshes_available, STUDIO_TO_REFERENCE_SIGN


class Scene:
    def __init__(self):
        # Studio 外观网格不随仓库分发。缺失时退回参考模型，并按关节轴比对得到的符号换算，
        # 保证每个关节在画面里的转动方向与 Studio 模型一致（外形、尺寸仍不同）。
        self.fallback=not studio_meshes_available()
        if self.fallback:
            self.model=SimRobot(ROOT/'models/dummy_reference.xml').model
            self.sign=STUDIO_TO_REFERENCE_SIGN.copy()
        else:
            self.model=SimRobot(ROOT/'models/dummy_studio_visual.xml').model
            self.sign=np.ones(6)
        self.data=mujoco.MjData(self.model)
        # Display only: forward kinematics, no dynamics or simulated force output.
        self.model.jnt_limited[:]=0
        self.ids=[self.model.jnt_qposadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,f'Joint{i}')] for i in range(1,7)]
        self.renderer=mujoco.Renderer(self.model,height=430,width=540)
        self.camera=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(self.camera)
        self.camera.lookat[:]=[0,-.06,.19]; self.camera.distance=.85
        self.camera.azimuth=135; self.camera.elevation=-18

    def frame(self,firmware_deg):
        # Studio-derived home-relative mapping; never used for hardware targets.
        self.data.qpos[self.ids]=self.sign*np.deg2rad(np.asarray(firmware_deg)-HOME)
        mujoco.mj_forward(self.model,self.data)
        self.renderer.update_scene(self.data,camera=self.camera)
        return self.renderer.render().copy()


class App:
    def __init__(self,root,auto_connect=True):
        self.root=root; root.title('Dummy · MuJoCo 实机同步上位机')
        root.geometry('1150x880'); root.minsize(1120,850)
        self.control=LiveController(ROOT/f'outputs/live_serial_{time.time_ns()}.jsonl')
        self.scene=Scene()
        if self.scene.fallback:
            root.title('Dummy · MuJoCo 实机同步上位机 —— 参考模型显示（Studio 外观文件缺失）')
            tk.Label(root,text='显示用的是参考模型：Studio 外观网格不在本机。关节转向已按 Studio 约定换算，'
                     '外形与尺寸不同，仅作示意。恢复方法见 models/README.md。',
                     bg='#fff3cd',fg='#6b4b00',anchor='w',padx=8,pady=4).pack(fill='x')
        self.generation=-1; self.setting=False; self.closing=False
        self.values=[tk.DoubleVar(value=v) for v in HOME]; self.readouts=[]; self.target_readouts=[]
        self.images=[]; self.last_q=HOME.copy(); self.last_draw=0.; self.drag=None
        style=ttk.Style(); style.theme_use('clam'); style.configure('TButton',padding=6)
        toolbar=ttk.Frame(root,padding=8); toolbar.pack(fill='x')
        ttk.Button(toolbar,text='连接 / 重新连接（只读）',command=self.control.connect).pack(side='left')
        self.arm_button=ttk.Button(toolbar,text='开启实时跟随',command=self.arm); self.arm_button.pack(side='left',padx=8)
        tk.Button(toolbar,text='停止跟随  [空格]',bg='#af3038',fg='white',command=self.stop,padx=15,pady=6).pack(side='left')
        ttk.Button(toolbar,text='目标回到当前实机',command=self.reset_target).pack(side='left',padx=8)
        self.home_button=ttk.Button(toolbar,text='直立复位',command=lambda:self.go_pose('home')); self.home_button.pack(side='left',padx=3)
        self.fold_button=ttk.Button(toolbar,text='折叠',command=lambda:self.go_pose('fold')); self.fold_button.pack(side='left',padx=3)
        ttk.Button(toolbar,text='断开',command=self.control.close).pack(side='right')
        options=ttk.Frame(root,padding=(8,2)); options.pack(fill='x')
        self.speed=tk.DoubleVar(value=5); self.speed_text=tk.StringVar(value='跟随速度：5.0°/秒')
        ttk.Label(options,textvariable=self.speed_text,width=25).pack(side='left')
        ttk.Scale(options,from_=1,to=20,variable=self.speed,command=self.change_speed,length=260).pack(side='left')
        ttk.Label(options,text='  上位机目标速率；实际速度受反馈与控制器限制').pack(side='left')
        auto=ttk.Frame(root,padding=(8,2)); auto.pack(fill='x')
        self.auto_button=ttk.Button(auto,text='启动六轴全范围线性往复',command=self.start_auto)
        self.auto_button.pack(side='left',padx=2)
        self.auto_result=tk.StringVar(value='各轴覆盖软件上下限；持续运行，按停止结束')
        ttk.Label(auto,textvariable=self.auto_result).pack(side='left')
        self.status=tk.StringVar(value='初始化 MuJoCo…')
        ttk.Label(root,textvariable=self.status,font=('Microsoft YaHei UI',11),padding=7).pack(fill='x')
        frames=ttk.Frame(root); frames.pack(fill='x',padx=8)
        self.views=[]
        for title in ('实机反馈姿态 · 原生 USB 角度','目标预览 · 滑块设定角度'):
            f=ttk.LabelFrame(frames,text=title,padding=2); f.pack(side='left',padx=4)
            label=ttk.Label(f); label.pack(); self.views.append(label)
            label.bind('<ButtonPress-1>',self.drag_start); label.bind('<B1-Motion>',self.drag_move)
            label.bind('<MouseWheel>',self.zoom)
        table=ttk.Frame(root,padding=8); table.pack(fill='x')
        for col,title in enumerate(('关节','反馈 °','目标 °','拖动调整目标（角度制）')):
            ttk.Label(table,text=title).grid(row=0,column=col,sticky='w',padx=8)
        self.sliders=[]
        for i in range(6):
            ttk.Label(table,text=f'J{i+1}',width=5).grid(row=i+1,column=0,padx=8)
            read=ttk.Label(table,text='—',width=11); read.grid(row=i+1,column=1,padx=8); self.readouts.append(read)
            out=ttk.Label(table,text=f'{HOME[i]:.2f}',width=10); out.grid(row=i+1,column=2,padx=8); self.target_readouts.append(out)
            slider=ttk.Scale(table,from_=LOWER[i],to=UPPER[i],variable=self.values[i],command=self.changed,length=740)
            slider.grid(row=i+1,column=3,sticky='ew',pady=3); self.sliders.append(slider)
        ttk.Label(root,text='拖动图像旋转视角，滚轮缩放。开启跟随后，拖动滑块即发往实体；自动测试期间锁定手动滑块。',padding=(12,3)).pack(anchor='w')
        ttk.Label(root,text='外观关节映射待校准；不用于碰撞判断。显示的是固件缓存角度，非带传感器时间戳的反馈。软件停止不能替代实体停止。',foreground='#8b5318',padding=(12,3)).pack(anchor='w')
        root.bind('<space>',lambda event:self.stop()); root.bind('<Escape>',lambda event:self.stop())
        root.protocol('WM_DELETE_WINDOW',self.close)
        if auto_connect: self.control.connect()
        self.tick()

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
    def zoom(self,event): self.scene.camera.distance=np.clip(self.scene.camera.distance*(.9 if event.delta>0 else 1.1),.3,2.)

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
        self.scene.renderer.close(); self.root.destroy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--smoke-test',action='store_true',help='Render home/fold poses to PNG without opening hardware')
    args=parser.parse_args()
    if args.smoke_test:
        scene=Scene()
        for name,q in [('home',HOME),('fold',[0,-75,180,0,0,0])]:
            Image.fromarray(scene.frame(q)).save(ROOT/f'outputs/live_{name}_preview.png')
        scene.renderer.close(); print('RENDER_OK'); return
    root=tk.Tk(); App(root,auto_connect=not args.offline); root.mainloop()


if __name__=='__main__': main()
