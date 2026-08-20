# AstrBot V4.0.5 可选运行组件

这里保存月社妃 V4.0.5 的 Skill 内存索引与 runner 接入。它们用于提高多轮身份、知情、物理状态和资料读取的可靠性，不是安装 Persona 与 Skill 的必要条件，也不是通用 AstrBot 插件。

## 文件

- `runner/yueshefei_skill_index.py`：读取并验证 `skill/resource_catalog.json`，把正式资料切分为只读内存片段。
- `runner/tool_loop_agent_runner.py`：在主生成前合并消息作者、V2 状态、持续事实和 Skill 语义需求，并将相关片段注入同一次主请求。

## 工作流程

```text
消息作者语义解析 + Skill 独立路由
→ 确定当前消息作者
→ V2 当前状态 + 持续事实
→ 合并两路资料需求
→ 内存索引选择完整相关片段
→ 注入主请求
→ 一次角色生成
```

两路中任一路认为需要资料，就会进入预取；两路都认为常驻内核足够时才跳过。实现不依赖人名、地点、动作、否定或解除词表。

## 怎样证明 Skill 真正进入

“路由请求成功”不等于“资料已参与回答”。部署验证时必须在同一轮日志中检查：

- `Skill prefetch selected` 有非空资源列表；
- `chars` 大于 0；
- `trace` 中有实际文件路径、SHA-256、行号与片段字符数；
- 随后出现 `Yueshefei Skill prefetch` 注入记录；
- 再随后是同一轮主请求与完整角色回复。

如果只有路由日志、没有片段 trace 或注入记录，应判定 Skill 接入失败。

## 兼容性与安装警告

该快照在 AstrBot `4.27.2` 上验证。runner 属于框架内部文件补丁，不同版本的内部接口可能变化。部署前必须保存原文件、核对导入路径并先在测试环境编译：

```text
python -m py_compile components/runner/tool_loop_agent_runner.py components/runner/yueshefei_skill_index.py
```

恢复时 runner 与索引必须成对部署，并保证完整 `skill/` 与 `resource_catalog.json` 同版本。若索引验证失败，不应把不完整读取当作成功。

组件不包含服务器地址、密码、API Key、状态数据库、真实会话或私人测试文本。
