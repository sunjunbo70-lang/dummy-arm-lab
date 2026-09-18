# 主工作区迁移

用户明确指定 D:\VLA 为唯一主工作区。原整理成果747个文件复制后逐一比对SHA-256，原始上游目录未修改。

copy_manifest.json 是迁移时的文件快照；后续对README、打包工具、交接文档的修改不应拿它当作最新版本校验。

重新创建 D:\VLA\.venv-loop 并通过 pip 安装 requirements/modeling.txt，未直接复制旧虚拟环境。新环境中 pip check、35项软件测试、117个证据文件校验、兼容与分类入口的离线渲染、参考策略240步仿真均通过。见 verification.json 和 rollout.summary.json。未发送实机运动指令。

后续开发、实验和打包都在 D:\VLA；旧 C 盘副本只用于回退。Python基础解释器仍是本机已安装的运行时；跨电脑迁移仍需重新安装Python和建立环境。
