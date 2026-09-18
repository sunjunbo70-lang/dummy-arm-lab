"""Read scene configuration only. Never execute Unity or access robot ports."""
from pathlib import Path
import json
import re
import UnityPy

src=Path(r'D:\VLA\_archive\dummy_v2\upstream\dummy-v2-ren-archive\3.Software\DummyStudio\DummyStudio_Data')
out=Path(__file__).resolve().parents[2]/'outputs/studio_scene_config.json'
records=[]
for name in ('level0','sharedassets0.assets','globalgamemanagers.assets'):
    env=UnityPy.load(str(src/name))
    for obj in env.objects:
        if obj.type.name != 'MonoBehaviour': continue
        raw=obj.get_raw_data()
        strings=[s.decode('ascii') for s in re.findall(rb'[\x20-\x7e]{4,}',raw)]
        if any(any(k in s for k in ('START','STOP','GETJPOS','CMDMODE','Write','BaudRate')) for s in strings):
            records.append(dict(file=name,id=obj.path_id,raw_strings=strings))
        try:
            data=obj.parse_as_dict()
        except Exception as exc:
            raw=obj.get_raw_data()
            strings=[s.decode('ascii') for s in re.findall(rb'[\x20-\x7e]{4,}',raw)]
            if any(any(k in s for k in ('START','STOP','GETJPOS','CMDMODE','Write','BaudRate')) for s in strings):
                records.append(dict(file=name,id=obj.path_id,strings=strings,error=str(exc)))
            continue
        text=json.dumps(data,ensure_ascii=False,default=str)
        if any(k in text for k in ('START','STOP','GETJPOS','CMDMODE','Write','BaudRate','minValue','maxValue')):
            records.append(dict(file=name,id=obj.path_id,data=data))
out.write_text(json.dumps(records,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
print(json.dumps(records,ensure_ascii=False,indent=2,default=str))
