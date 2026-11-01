# NInfer dense heads TP 验收记录（2026-10-30）

## 当前集成：2026-11-01

本次在 master `f2d28bf` 上整合 dense heads TP、checkpoint 清理与结构化 InputReuse。
NInfer 仍固定 `d0da39796697`，27 层补丁逐层重放并与实际原生工作树一致，prepared tree
为 `1e7a5d621c341f2c257b81b9a5632dd19b17b07f`。本轮 worker SHA256：
`6ba4c564aedb3bf27a872a7a5a9f3870f392ded33b0e54f67b5d8593816d1c03`。
详细数值和限制见 [本轮验收摘要](benchmarks/ninfer-heads-merge-20261101.json)。

新增行为由 [checkpoint policy](checkpoint-policy.md#ninfer-input-reuse-and-template-safe-fallback)
统一描述：模板安全回退点继续保留；结构化输入边界另存完整 KV/GDN 镜像；普通工具续接仍为
append + resident-only Pressure。快照额外重算上限默认 0，16 是可选值，未改为默认。

源码审计核对了可选捕获成功前的旧来源保留、完整状态/Query 保护、已完成请求的清理、
多 rank payload 身份以及启动能力门控。没有发现新的产品阻塞缺陷。合并时保留了 master 的
Radiance 图片解码门控、GPU 展示及 opaque packed KV 规则，并补测 packed KV 与 rank/head
身份共存、非法 token stride 和重叠 head 范围拒绝。

本轮验证：4 个共享/native 主机目标；启动计划 29 通过、42 个平台用例跳过；控制台 61、
Radiance launcher 9；8 组 JavaScript 检查及 Windows 参数矩阵。最终二进制完成 27B heads
MTP 的 T0/T16 短测 12 请求、Query/SSD 24 请求，均有原生恢复、真实冷读、取消和后续恢复
证据；4B 双物理卡 InputReuse/MTP 两档共 14 请求通过；IQ3_S 主模型/Q3 DFlash2 双卡
Query/Pressure 回归通过。普通 document-first/question 测试保持分开请求。

27B 双卡性能采用同一 worker、artifact、GPU 顺序（5060 Ti / 5050 Laptop）、3:1 heads、
INT8 KV、MTP2、B768/R1024、prefill128；499 输入、128 输出，每档先预热一对，再测三对
cold/repeat 请求。计时关闭 transfer 验证与 trace；T0 先于 T16，原始三轮均保留。

| 中位数 | T0（默认） | T16 |
|---|---:|---:|
| 冷请求 prefill | 1.598690 s | 1.474374 s |
| 重复请求 prefill | 0.095392 s | 0.240247 s |
| 重复请求原生 TTFT | 0.096604 s | 0.242217 s |
| 重复请求实际重算 tokens | 1 | 7 |
| 冷请求墙钟 | 4.140 s | 4.015 s |
| 重复请求墙钟 | 2.625 s | 2.765 s |
| 每对请求墙钟中位数 | 6.750 s | 6.765 s |
| 整卡显存采样峰值（5060 Ti / 5050） | 9830 / 2846 MiB | 9830 / 2846 MiB |

T16 冷 prefill 较快，但重复请求明显变慢，成对总耗时也未改善；额外 tokens 上限不是
延迟或完整 state 重建成本上限。本样本不支持开启 T16 默认值、普遍提速或最优阈值结论。
InputReuse 可额外保留完整 Host StateImage 与元数据，内存开销依模型而定；极端 Query 与
InputReuse 重叠时的峰值内存本轮未另行测量。

未计为通过的尝试也保留：27B 原生 InputReuse 夹具在前六步边界/恢复检查后，因该 artifact
未产生夹具所要求的 4B 模板包装重写而停止；这不是一次完整通过，也没有修改模板来满足
断言。4B 双卡验证覆盖夹具原定的正向路径。旧 Q4 DFlash2 artifact 已不在本地，本轮明确
使用现有 Q3 artifact。未新增 native allocation/copy 故障注入、完整 NVFP4 权重模型、
其他架构或 3–8 卡验证；本轮不是广泛语义质量评测。未替换已安装运行包。

以下 2026-10-30 记录及其中的“未提交/未发布”、旧 worker 性能、数值和已知文本差异均为
历史事实，不应当作本轮重新测量。当前集成结果以本节及验收摘要为准。

## 2026-10-30 历史验收


2026-10-30 二次审计后的两项本地回归现已修复：rank 0 的实际 TP 分片应用统一 NVFP4 精度策略；采样页面质量在生产路径按页面合并，容量计费包含擦除零项后保留的存储，32 MiB 限额不变。修复后树为 `44c08b83d840057e67f13f6369910d75a682181b`，worker SHA256 为 `db22edc5501ade11b4662c75ac32e646d19489ecfe826c0821ad49bfc2dc3347`。

本轮增量验证在 `fix-regressions-20261030/validation.json`：真实 artifact→Model→Parameters 的 8 种策略/不等比分片及普通对照通过，同一测试链接旧 runtime 会正确失败；生产聚合器完成 133120-token 单/多 rank 主机回归并与独立 Pressure oracle 一致；4B 单卡/双卡和 27B IQ3、PTQ1、T5 双卡 MTP 的 Query/Pressure、工具续接和取消验证通过。新候选使用 `candidate-package-regressions-fixed/`，包验证记录见当前 `completion.json`。原候选及审计反例保留，没有提交、发布或替换默认包。

合成 NVFP4 测试使用单张 5060 Ti 上的两个逻辑 rank，未执行完整 NVFP4 主模型推理；INT8 小形状验证的是 A16 fallback。长窗口反例本轮在真实 host 库验证，未做同长度 GPU 请求。rank 0 的 PTQ1 FP32 partial 现在与其他 rank 一致使用 A16 fallback，本轮未测量该条件的性能代价。初次合成夹具的 MRoPE 与 activation-divisor 错误已修正，失败记录保留。**以下是审计前 worker `98d270f2…` 的广泛验收和性能历史，不是修复后重新测量；其中已知 DFlash2 文本差异及性能代价仍应保留。**

本轮将 NInfer 固定到 `2026-10-29.2` / `d0da39796697b79fd08fc570e1781d75943dddb1`，迁移原有 24 层补丁并增加 `heads-tp` 第 25 层。KVMem 基线为 `ea156ea5b63d32febf991aa816bd0f5322de4f8b`；交付前官方 master 核对仍为该提交。成果是本地源码和独立候选包，未提交、合并、发布或替换默认运行包。

## 行为和边界

TP 仍需显式启用，固定使用 heads，不开放 positions。自动权重和按 GPU UUID 顺序指定的相对权重均可用；实际 KV/GDN/FFN 切分由上游合法边界取整，不能把输入权重当作实际 head 数或显存占比。`/props.execution` 和监控显示实际分片。启动/导出要求 worker 报告整数 `heads_tensor_parallel_revision >= 1`，与 StageLayers 互斥。

Query/MeanQ 统计按真实 head 范围汇集；attention page mass 按 head 数加权。KV payload descriptor、StateImage 身份和事务覆盖所有实际 rank，MTP/drafter 继续归 rank 0。启动规划和自动 TP 预算使用同一统计分配规则。普通工具续接保持 append + resident-only Pressure，不因缓存了 Q 而冷召回。

PTQ1/T5 的 FP32 partial 输出和合法非均匀切分、NVFP4 A16/A4/INT8 路由已经适配。此次没有增加 MoE、positions、视频、新视觉能力或运行中重分片。

## 实际验证范围

环境为 Windows、VS 2022 / MSVC 14.44、CUDA 13.2、Release、`sm_120a`。固定依赖为 `ninfer-x64-windows-c7684b1d96347de8`。单卡优先使用 RTX 5060 Ti；双卡顺序为：

- RTX 5060 Ti：`GPU-A`
- RTX 5050 Laptop：`GPU-B`

每组运行前核实空闲。两卡原生诊断报告无 peer access，实际通过 host memory 中转。没有跨架构、3–8 卡或 Linux 实测。

| 范围 | 已取得的证据 |
|---|---|
| 主机契约 | 不等比分片、无 attention head rank、预算、payload 身份、参数拒绝、artifact 选择、MTP Graph profile |
| 独立数值 | GQA4/6/8 attention、head 加权 page mass、GDN replay、PTQ1/T5 Linear/SwiGLU、NVFP4 A16/A4/INT8、FP32 strided SiLU、成对 row split；采用独立 FP32/FP64/精确编解码 oracle，包含 Graph |
| 普通 heads | 4B Q4_K_M + INT8 Graph；27B GSQ-RCO IQ3_S + RK8V4 eager，手动 3:1 |
| MTP heads | 27B IQ3_S + INT8 自动分片、NVFP4 KV 手动 3:1；PTQ1/T5 27B 真模型 + INT8，MTP2 Graph |
| DFlash2 heads | 27B 内置 DFlash2，INT8/RK8V4/NVFP4 KV 长上下文；额外单卡、双卡 Graph/eager、双并发短请求 |
| 原有模式 | 27B layer pipeline 40+24 层完整请求回归；新旧单卡的普通解码性能请求；新版单卡 DFlash2 |
| Query/Pressure/replay | 文档先发、同会话单独问题；实际 prefix 复用和短 Query span；工具续接无冷选择，取消后恢复、会话切换、resident 超 B、新 Q 捕获 |
| 状态与 SSD | 物理双卡 27B RAM、INT8/RK8V4/NVFP4 SSD；真实读写、clear/cleanup、accepted-prefix 1–5 token 和跨 reserve；跨 owner fork 冷读失败/取消、父历史保留 |
| 启动入口 | Python launcher 18 项、local-console 53 项、Windows argv/非法组合及浏览器 launcher/monitor 测试 |

49 项打包前检查已归档。数值与主机测试保留各自测试二进制身份；后续受影响的 SiLU 和 peer fixture 已重跑。所有实际 HTTP 和有效性能测量对应最终 worker：

`98d270f21b17f4691ab411d6e0864b7770855a783a33f7f1a1f7cf7b8c315fbd`

独立候选包还通过了 10 项真实浏览器验收：413 个打包文件与校验清单匹配，包内 Python/worker 启动，默认 TP 关闭，UUID 顺序和 3:1 权重预览，启动后配置锁定，4B 的真实 SSE 回答，监控服务来源的 KV `[0,3)`/`[3,4)`、GDN 12+4、FFN 6912+2304，以及界面停止后服务端口关闭、控制台仍可用。结果为 `package-console-acceptance-complete/result.json`，非 API 替代流程，浏览器错误为空；已目视检查聊天和实际分片截图。

小型 StateImage roundtrip 用同一 GPU 的两个逻辑 rank；真实跨物理卡证据来自另列的 27B 状态测试。SM70 专用 panel 路径在本机跳过。没有完整 NVFP4 主权重 artifact，因此不能把 NVFP4 KV 端到端通过或 NVFP4 权重算子通过写成完整 NVFP4 主模型已验收。未做本轮视觉端到端或统计性语义质量评估。

## 受控性能：收益和代价

同一 27B GSQ-RCO IQ3_S artifact，INT8 KV，普通解码，Graph，device profile off，chunk=128，B=768，R=1024；499 prompt tokens、128 output tokens、temperature=0、cache=0。每组预热一次后测三轮；没有并行编译或其他 GPU 任务。下表是中位数，括号为三轮最小–最大值。

| 执行方式 | PP tok/s | TG tok/s | 原生 TTFT 秒 | 采样峰值显存 MiB（5060 Ti / 5050） |
|---|---:|---:|---:|---:|
| 本地 v0.19.1 包单卡 | 509.6（502.7–538.0） | 31.38（31.01–31.42） | 0.980（0.929–0.994） | 11696 / 0 |
| 新候选单卡 | 511.7（501.3–537.1） | 30.96（30.963–30.976） | 0.977（0.930–0.997） | 11696 / 0 |
| 新候选 heads 3:1 | 318.8（318.4–323.4） | 33.08（33.01–33.09） | 1.566（1.544–1.569） | 9128 / 2846 |
| 同候选 heads，关闭 KVMem 的诊断 | 368.4（368.3–368.6） | 34.26（34.13–34.31） | 1.356（1.355–1.356） | 9256 / 2888 |

新旧单卡是相同 5060 Ti 的比较：PP 中位数 +0.4%，TG -1.3%。TG 的小幅下降如实保留，没有根据三轮数据断言其成因或将其排除为噪声。

新候选双卡相对新单卡 TG +6.8%，但 PP -37.7%、TTFT +60.4%；完整请求墙钟中位数 5.500 秒，相对单卡 5.391 秒更慢。主卡显存下降 2568 MiB，但两卡合计采样值略增。TP 用了额外硬件，不能称作同硬件提速，也不适合据此推荐所有模型开启。默认继续关闭 TP，优先把它视为容量分摊选项。

关闭 KVMem 的同候选诊断仍呈现明显的 TP 预填充代价；KVMem 开启相对此诊断 PP -13.5%、TG -3.5%。两者的 cache/驻留规划不同，这不是全部补丁成本的精确分解，也不是未修改上游的比较。未取得相同 Windows 工具链的原版上游运行对照。没有将本短提示结果外推到长上下文、MTP、DFlash2 或其他格式。

## 修复过的失败与仍有的差异

- 迁移中的 GQA4 small-T 路由错误导致缺失实例化/链接失败，已按完整 Q/KV 几何修正并通过数值测试；不支持的 GQA4 positions 在变更状态前拒绝。
- heads 的 FP32 gate/up 切片在多 token 下不连续，原 SiLU 路径拒绝；已扩展既有 strided kernel 至 FP32，保留连续快速路径。数值、Graph 和 PTQ1/T5 真模型重测通过。初次 T5 无栈快速失败单独不能确定根因，原记录保留。
- Windows 的 Linux sysfs fixture 使用非法冒号路径；已分平台保留相应测试。真实双卡通信、host fallback 和 Graph 验证通过，未声称在 Windows 测过 Linux sysfs 解析。
- fork fixture 的 2 GiB SSD 配额不足，导致取消时选中热状态；仅测试改为按需 8 GiB 上限并等待实际冷状态，保留且加强取消断言。重测实际采样 payload 最大 3,079,086,880 字节，冷读错误、取消、未发布 hit、资源释放和父历史保留全通过；生产策略未改。
- harness 的 PowerShell 分隔参数、普通解码的多余 draft 参数、GCC 硬件控制程序同名识别，以及原生缓存导致的无效性能轮次均已修正。原无效记录保留；仅原生诊断组需要禁用 prefix reuse 后重测，三组原有效测量未替换。
- DFlash2 两条短提示中，一条在双卡 Graph/eager 下与普通解码文本不同，两种执行相互一致；单卡及双并发这两条均匹配。边界与数值证据没有显示错误，但尚无语义质量统计，不能宣称跨模式逐 token 相同。
- 包内 UI 验收脚本也修复了 PS5 参数默认目录、PATH 内多份 Node、默认值省略断言、运行生成的 Python 缓存影响严格清单、Chromium 消费后 SSE 取证以及未消费 HTTP 响应触发 Node 断言的问题。它们未要求修改候选产品源码；所有失败/中断尝试保留。最终完整闭环 exit 0，实际 UI 聊天和停止均通过，不将中途已通过的部分冒充完整验收。

## 源码、证据和回退

源码在 `ninfer-heads-20261030/repo`，独立原生源码在 `ninfer-heads-20261030/backend`。`backends/versions.json` 保存每层 patch hash/依赖/tree，25 层从固定上游干净索引重放并与实际源码核对通过；最终 prepared tree 为 `b68f3788da967c4f2df4593937a819b2c0f35cd7`。

工作区 `$ARTIFACTS/ninfer-heads-20261030/` 下保留 `native-validation.json`、`state-audit.md`、`validation/` 请求/响应/原生日志/性能数据、固定依赖锁、构建计划和失败重试记录。独立候选位于 `candidate-package/`，包内 `provenance/validation.json` 是打包前验收；实际 UI 验收另存 `package-console-acceptance-complete/`，总体索引为 `completion.json`。

原 `release-v0.19.1-rtx50-20261029/package/kvmem-v0.19.1-windows-x64-nvidia-50-ninfer` 保持原样。使用旧包即可回退；不能将新旧进程的 KV/checkpoint 文件互换。
