# 2026-09-25 实验版本化整理

用户明确要求将所有实验资料移入experiments，撤销根launchers。本次为软件整理，不连接硬件，不重训，不删除失败记录。

moves.json列出94个搬移入口；files_before.json是迁移前13299个文件清单；verification.json证明所有文件存在、未改文件大小一致、13个选中模型SHA256相同、恢复历史链可读、修改源码语法正确。original_texts.zip保存路径替换前原件，changed_texts.json列出机械路径更新文件。JSONL、PT、NPZ原始数值未改，原来冻结的源代码哈希仅描述冻结时版本，迁移后的路径修改不冒充冻结源码。

根目录launchers/outputs/design及D:/VLA/experment design的资料已迁入；旧空目录已移除。共享源码dummy_loop、工具tools、模型models仍保留。模型设计资料在00_initial_debug/design，相关实验证据在records。

源码和文档路径更新；所有29个cmd归入版本scripts，修正仓库根目录相对层级。没有执行训练或硬件入口。3项入口契约测试、3项回放/日志回归测试通过，最终报告可在新目录生成。文件定位可通过moves.json查旧路径的新位置。

备份说明：大型runs及original_texts.zip不进Git，但在本机保留。跨电脑必须另行复制runs和受许可限制的本地模型，并重新创建Python环境。迁移不代表新电脑已验收。

回滚应先停用当前实验：根据moves.json逆序移动回原路径，按original_texts.zip恢复路径修改前文本。不得覆盖后来产生的新文件；若目标已存在，先人工核对。不要直接递归删除版本目录。
