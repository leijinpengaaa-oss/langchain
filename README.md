# 知行 · 知识与工单助手

基于 **LangGraph、FastAPI、千问和SQLite** 的本地AI应用：从文档中回答问题并展示原文依据，将故障整理成可编辑工单，经人工审批后保存，支持会话持久化与跨重启恢复。

应用代码位于 `apps/knowledge_desk/`。仓库中的 `libs/`、框架示例和原有文档来自上游 [LangGraph](https://github.com/langchain-ai/langgraph)，本项目没有修改其核心库。上游说明保留在 [README.langgraph.md](README.langgraph.md)，许可证见 [LICENSE](LICENSE)。

![真实千问回答与可展开原文摘录](apps/knowledge_desk/reports/answer-evidence-v3.jpg)

## 主要能力

- **文档RAG问答**：上传Markdown、TXT或含文字层的PDF；同义词BM25与千问向量混合召回，RRF融合及千问候选重排，支持纯语义、无重排对照和接口故障降级。
- **可核验来源**：引用本次检索片段，核对连续原文摘录；无依据时拒答。摘录存在不能保证回答语义正确，另以分项评测与人工复核检查。
- **人工审核与恢复**：LangGraph状态图、`interrupt`与SQLite检查点；编辑后确认工单，取消不写入，待审任务重启后可继续。
- **幂等与缓存**：请求、会话和工单操作三层重复保护；按空间及模型配置隔离的持久向量/重排缓存。
- **中文演示界面**：对话、知识库、检索效果对比、工单中心及运行记录；SSE展示节点进度，记录调用用量与耗时。

## 本次验证

48项自动化测试通过。132条自编排序对照中，原验证集18条正例的Hit@1由RRF的66.7%提升到重排的94.4%，恢复3条FAQ回退。24条问题的两组实际问答中，12条可回答题的事实规则检查由8/12提升到12/12，两组12条无依据题均正确拒答。

这些是自编、小样本数据，原验证集已在前轮查看，不是独立盲测或整体业务准确率。模型辅助评审存在偏差，原始失败与争议结果完整保留。本次24题图执行P95由5.797秒增加到11.364秒；使用聊天模型重排带来额外延迟与用量。

- [完整排序与答案质量报告](apps/knowledge_desk/RANKING_AND_ANSWER_QUALITY.md)
- [两组逐题原文与答案对照](apps/knowledge_desk/reports/answer-quality-v3.html)
- [架构、检查点与幂等设计](apps/knowledge_desk/ARCHITECTURE.md)
- [项目经历与面试说明](apps/knowledge_desk/PROJECT_NOTES.md)

## 快速开始

推荐Python 3.11。以下命令在仓库根目录的PowerShell中执行：

```powershell
git clone https://github.com/leijinpengaaa-oss/langchain.git
cd langchain
python -m venv .venv-demo
.\.venv-demo\Scripts\python.exe -m pip install -e ./libs/checkpoint -e ./libs/sdk-py -e ./libs/prebuilt -e ./libs/langgraph -e ./libs/checkpoint-sqlite
.\.venv-demo\Scripts\python.exe -m pip install -r ./apps/knowledge_desk/requirements-dev.txt
Copy-Item .env.example .env
```

在`.env`中填入自己的`DASHSCOPE_API_KEY`，然后启动：

```powershell
.\.venv-demo\Scripts\python.exe -X utf8 .\run_knowledge_desk.py
```

打开 <http://127.0.0.1:8765/>。首次访问会导入3份虚构星舟资料。没有密钥时可使用明确标识的离线示例：

```powershell
.\.venv-demo\Scripts\python.exe -X utf8 .\run_knowledge_desk.py --offline
```

真实模式会把问题、会话与候选资料发送到配置的千问接口并产生API用量。运行数据库、Cookie签名密钥、`.env`、虚拟环境与缓存不会提交到仓库。演示面向单进程本地应用，当前没有账号认证或分布式任务协调。

## 检查与评测

```powershell
.\.venv-demo\Scripts\python.exe -m ruff check apps/knowledge_desk run_knowledge_desk.py
.\.venv-demo\Scripts\python.exe -m ruff format --check apps/knowledge_desk run_knowledge_desk.py
.\.venv-demo\Scripts\python.exe -m pytest apps/knowledge_desk/tests -q
# 以下两项真实调用API：
.\.venv-demo\Scripts\python.exe -X utf8 -m apps.knowledge_desk.ranking_benchmark
.\.venv-demo\Scripts\python.exe -X utf8 -m apps.knowledge_desk.answer_quality
# 生成逐题HTML与人工评分模板，无API调用：
.\.venv-demo\Scripts\python.exe -X utf8 -m apps.knowledge_desk.quality_review
```

更多配置、演示步骤及复现说明见 [应用README](apps/knowledge_desk/README.md)。上游框架的发布、Pages及定时工作流未纳入本仓库上传范围。
