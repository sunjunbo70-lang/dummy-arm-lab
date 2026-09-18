"""Optional D435f check. Lists supported streams, requests no robot connection."""
import json
import argparse

p=argparse.ArgumentParser()
p.add_argument('--frames',type=int,default=0,help='0 lists capabilities; >0 verifies default RGB-D streams')
a=p.parse_args()
if not 0<=a.frames<=300: p.error('frames must be 0..300')
import pyrealsense2 as rs
devices=rs.context().query_devices()
for device in devices:
    profiles=[]
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() not in (rs.stream.color,rs.stream.depth): continue
            if profile.is_video_stream_profile():
                v=profile.as_video_stream_profile()
                profiles.append({'stream':str(v.stream_type()),'format':str(v.format()),'width':v.width(),'height':v.height(),'fps':v.fps()})
    print(json.dumps({'device':device.get_info(rs.camera_info.name),'profiles':profiles},indent=2))
if a.frames:
    if len(devices)!=1: raise RuntimeError('Exactly one camera required for this probe')
    pipeline=rs.pipeline(); config=rs.config()
    config.enable_stream(rs.stream.depth); config.enable_stream(rs.stream.color)
    active=pipeline.start(config)
    try:
        print(json.dumps({'depth_scale':active.get_device().first_depth_sensor().get_depth_scale()}))
        for _ in range(a.frames):
            frames=pipeline.wait_for_frames(5000)
            depth,color=frames.get_depth_frame(),frames.get_color_frame()
            if not depth or not color: raise RuntimeError('Missing depth or color frame')
            print(json.dumps({'depth_frame':depth.get_frame_number(),'color_frame':color.get_frame_number(),
                'depth_timestamp_ms':depth.get_timestamp(),'color_timestamp_ms':color.get_timestamp(),
                'depth_clock':str(depth.get_frame_timestamp_domain()),'color_clock':str(color.get_frame_timestamp_domain())}))
    finally: pipeline.stop()
