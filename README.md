# OpenCode 会话头注入（maibot.opencode-session-header）

给 MaiBot 用的插件：当模型请求指向 **`opencode.ai`** 网关时，自动注入稳定的
**`X-OpenCode-Session`** 请求头，避免网关返回 `400 MissingSessionID`；
其他 Provider 完全不受影响。

会话令牌由「常见会话头」（`Session-Id`、`X-Session-Id`、`X-Client-Request-Id` …）
经 **SHA-256 截断哈希**派生，原始会话 ID 永不出网；找不到会话头时使用
一份持久化的随机种子兜底，保证同一个部署始终拿到同一个令牌。

---

## 1. 它解决什么问题

OpenCode 网关（`https://opencode.ai/zen/...`）从某个版本起要求每个请求携带会话标识，
缺失时直接拒绝：

```
HTTP 400  {"error": {"message": "MissingSessionID", ...}}
```

MaiBot 内置的 `openai` / `openai_responses` 客户端不会发送这类会话头，于是所有
opencode 模型请求都会失败。本插件接管这些请求，在转发前补上会话头。

## 2. 工作原理

MaiBot 允许插件注册自己的 **LLM Provider**（对应模型配置里的 `client_type`），
这是插件唯一能影响出网请求的官方扩展点（内置 `client_type` 不允许被插件覆盖）。
本插件因此：

1. 注册 `client_type = "opencode-session"` 的 Provider，接管指向它的模型请求；
2. 读取本次请求的 `api_provider.base_url`，判断是否命中 `opencode.ai`（含子域）；
3. 命中时按优先级解析会话种子，做 SHA-256 截断哈希，注入会话头；
4. 未命中时**不添加任何请求头**，请求体原样转发 —— 与内置客户端行为一致；
5. 按 `[wire].wire_protocol` 选择线上协议（`/chat/completions` 或 `/responses`），
   并把网关响应还原成宿主可消费的结构（正文 / 推理内容 / 工具调用 / usage）。

> 只影响你把 `client_type` 指向本插件的那些模型。使用内置 `openai`、
> `openai_responses`、`gemini` 的模型、以及其他 Provider，一条请求都不会被改动。

## 3. 安装与启用

插件目录：`data/MaiMBot/plugins/opencode-session-header/`（本机为
`/root/data/MaiMBot/plugins/opencode-session-header/`）。

1. 把整个目录放到 `plugins/` 下（`_manifest.json`、`plugin.py` 必需）；
2. 打开 WebUI（本机 <http://127.0.0.1:18001>）→ 插件管理 → 找到
   **OpenCode 会话头注入** → 启用；
3. 首次加载后，插件目录内会自动生成 `config.toml`（**不要提交**，已在 `.gitignore` 中）。

## 4. 关键一步：把模型指向本插件

插件注册的 `client_type` 是 `opencode-session`。**必须**把它填到 opencode 服务商上，
插件才会生效。两种方式：

**WebUI（推荐）**：设置 → 模型 → 编辑 opencode 的 API 服务商 →
「客户端类型」选择 `opencode-session`。

**手改 `config/model_config.toml`**：

```toml
[[api_providers]]
name = "opencode"
base_url = "https://opencode.ai/zen/v1"
api_key = "sk-..."          # 你自己的密钥
client_type = "opencode-session"   # ← 关键：默认是 "openai"
```

保存后重启或热重载模型配置即可。

> 改回 `openai` 就等于停用注入（插件不再参与该服务商的任何请求）。

## 5. 配置项（WebUI 可编辑，热重载生效）

> 从 1.1.0 起新增 `[wire]` 配置节。旧版 `config.toml` 会在插件加载时**自动迁移**：
> Runner 用「补充缺失键」的方式合并默认值，保留你手写的注释与原值。

### `[plugin]`

| 键 | 默认值 | 说明 |
|---|---|---|
| `enabled` | `true` | 是否注入会话头。关闭后请求照常转发，只是不加头 |
| `config_version` | `1.1.0` | 配置版本，勿手改（与插件版本号无关，仅在配置结构变化时递增） |

### `[endpoint]`

| 键 | 默认值 | 说明 |
|---|---|---|
| `host_suffixes` | `["opencode.ai"]` | 视为 OpenCode 的域名后缀，自动覆盖子域 |
| `extra_base_url_prefixes` | `[]` | 额外的 URL 前缀，用于自建/中转 OpenCode 网关。**命中这些前缀的请求，会把完整请求体（含图片 base64）与鉴权头发往该地址**，请只填你自己信任的网关 |

### `[session_header]`

| 键 | 默认值 | 说明 |
|---|---|---|
| `header_name` | `"x-opencode-session"` | 注入的头名 |
| `extra_header_names` | `[]` | 可选：再注入同令牌的头，如 `x-session-affinity`、`x-client-request-id` |
| `token_length` | `8` | 截断长度，4–32 |
| `token_alphabet` | `"alphanumeric"` | `alphanumeric`（纯字母数字）或 `urlsafe` |
| `salt` | `""` | 可选盐值；修改后所有令牌都会变化 |
| `candidate_headers` | 见下 | 会话种子候选头，按优先级排列 |
| `session_id_env` | `""` | 可选：找不到会话头时读取的环境变量名。**只填会话标识类变量名**，不要填 API Key / 密钥类变量名——种子的熵会来自该值 |
| `verbose` | `false` | 记录每次注入的端点、种子来源与令牌。**日志里会出现令牌与端点本身**，排查完建议关掉并注意日志留存范围 |

`candidate_headers` 默认值（先命中者参与哈希）：

```
x-opencode-session, session-id, x-session-id, session_id, x-session-affinity,
x-client-request-id, x-request-id, conversation-id, x-conversation-id,
thread-id, x-thread-id, chat-id, x-chat-id
```

查找顺序：**服务商 `default_headers` → 模型 `extra_params.headers` → 请求 `extra_params.headers`**。
都没有时依次回退到 `session_id_env` 环境变量、最后是插件数据目录中的持久化随机种子
（`data/plugins/maibot.opencode-session-header/session_seeds.json`）。

### `[wire]`

| 键 | 默认值 | 说明 |
|---|---|---|
| `wire_protocol` | `"chat_completions"` | `chat_completions` / `responses` / `auto`，见下 |
| `responses_store` | `false` | 仅 `responses` 协议：是否让网关保存响应（`store`） |

## 5.1 协议选择（chat_completions / responses）

插件能用两种线上协议把请求送到网关，**端点识别与会话头注入逻辑完全一致**，只有请求/响应
的线格式不同：

| `wire_protocol` | 端点 | 行为 |
|---|---|---|
| `chat_completions`（默认） | `POST {base_url}/chat/completions` | 明确使用 Chat Completions |
| `responses` | `POST {base_url}/responses` | 明确使用 Responses API |
| `auto` | 先 chat，失败再 responses | 端点返回 404/405/501 时自动改用 responses，并按端点记住结论，后续不再重复试错 |

实测 `opencode.ai/zen/go/v1` **两个端点都支持**，所以默认值就够用：

```
POST /zen/go/v1/chat/completions -> 401 AuthError: Missing API key.
POST /zen/go/v1/responses        -> 401 AuthError: Missing API key.
```

（无鉴权探测；返回 AuthError 而非 404，说明路由与模型都被接受。）

**什么时候该选 `responses`**：你的中转网关只实现了 `/responses`，或者你确实需要
Responses 的结构化输出（`text.format`）。

**选 `responses` 时与内置客户端的差异**：主程序内置的 responses 客户端依赖
**provider replay fragment** 把推理项原样回传（`_can_replay_item()` 会校验 client_type、
端点指纹与模型名），插件这条路拿不到 replay —— `serialize_context_item_snapshot()`
明确不把 replay 放进快照，宿主反序列化时也不会还原。因此插件在 responses 模式下
**直接省略 reasoning 项**，只发 `function_call` + `function_call_output`。

这带来的实际影响比想象中小：

* 单轮问答、普通工具调用：**没有区别**；
* 多轮工具循环：丢掉的是**跨轮次的推理连续性**（质量层面），不是报错。
  已知会真报错的是另外两种写法 —— 回传 `rs_...` ID（`store=false` 下报
  `Item with id ... not found`）或回传残缺的 reasoning 项（报
  `reasoning was provided without its required following item`）。
  插件两种都不做，所以**不会因为这一项被网关拒绝**；
* 主程序内置客户端在 replay 作用域不匹配（换了 Provider / 端点 / 模型）时，
  走的也是同样的「丢弃 reasoning」降级路径。

结论保持不变：**没有特殊需求就用默认的 `chat_completions`**，那是零差异的路径。

## 6. 命令与工具

| 类型 | 名称 | 用途 |
|---|---|---|
| 命令 | `/opencode_session`（别名 `/oc_session`） | 查看注入开关、匹配域名、当前协议、令牌预览 |
| 命令 | `/opencode_session_reset`（别名 `/oc_session_reset`） | 清空兜底种子，下次请求重新生成 |
| 工具 | `opencode_session_token` | 让模型计算某个 Base URL 会用哪个协议、注入什么头，用于排障 |

## 7. 隐私说明

* 出网的永远是 **SHA-256 截断哈希**，不是原始会话 ID；
* `session-` / `sess_` 一类前缀不参与哈希，避免同一会话换前缀后令牌漂移；
* 令牌长度默认 8 位（约 47 bit 熵），只用于网关侧的会话黏连，不可逆推；
* 增加 `salt` 可以让不同部署的令牌空间互不相干；
* **请求会发往哪里**：插件转发的是原本就要发给模型服务商的请求。请求体（含图片 base64）
  与鉴权头会发往该服务商 `base_url` 指向的地址；若在 `endpoint.extra_base_url_prefixes`
  里配置了额外前缀，命中这些前缀的请求同样会把完整请求体与鉴权头发往**该前缀指向的地址**。
  这是显式配置行为，请只在你自己信任的自建 / 中转网关上启用——除此之外，插件不会向任何
  其他地址发送数据；
* `session_id_env` 只应填写会话标识类变量名。若填了持有密钥的变量名，出网的虽然仍是单向
  哈希、不可逆推，但种子的熵会来自该密钥的值；
* `verbose` 日志会输出端点、种子来源与令牌本身，请按你所在部署的日志留存策略处理；
* 插件只读取配置里的 URL 与请求头，不读取、不上报任何聊天内容。

## 8. 排障

**仍然 400 MissingSessionID**
1. 确认该 API 服务商的 `client_type` 已是 `opencode-session`（见第 4 节）；
2. 打开 `[session_header].verbose`，看日志里是否出现 `OpenCode 会话头注入: ... matched=True`；
3. 用 `/opencode_session` 确认注入开关为「开启」、域名列表包含 `opencode.ai`；
4. 若是自建/中转网关，把地址前缀加进 `endpoint.extra_base_url_prefixes`。

**插件没出现在 WebUI**
`_manifest.json` 是严格模式，多一个自造字段就整体加载失败。查看日志中
`Manifest 校验失败` 的原文，或运行：

```bash
docker exec -w /MaiMBot maim-bot-core .venv/bin/python -c "
from pathlib import Path
from src.plugin_runtime.runner.manifest_validator import ManifestValidator
v = ManifestValidator(); m = v.load_from_plugin_path(Path('/MaiMBot/plugins/opencode-session-header'))
print('OK', m.id, m.version) if m else print('FAIL', v.errors)"
```

**改了配置没生效**
`config.toml` 由 Runner 生成与维护；在 WebUI 里改，或改完文件后在插件管理里重载。

**突然开始报鉴权错误**
本插件用自己的传输层发送请求，鉴权方式完全按服务商配置里的
`auth_type`（`bearer`/`header`/`query`/`none`）与 `api_key` 执行；
如果服务商用的是自定义头，请检查 `auth_header_name` 与 `auth_header_prefix`。

## 9. 已知限制

* 只处理 `response` 操作（对话），不支持 embedding / 语音转写；
* 不启用流式：`force_stream_mode` 会被忽略，结果与非流式调用一致，只是首字延迟略高；
* 宿主发送的是「Context Item」快照，插件按主程序规则投影成 `messages`（或 Responses 的
  `input`）；遇到未知 Item 类型会跳过而不是让请求失败；
* `responses` 协议下拿不到 provider replay fragment，插件会省略 reasoning 项，
  影响是跨工具轮次的推理连续性（质量层面），不会因此被网关拒绝，
  详见第 5.1 节。没有特殊需求请使用默认的 `chat_completions`。

## 10. 开发与测试

```bash
# 单元测试（标准库 unittest，主机上直接跑）
python3 -m unittest discover -s tests -v

# 插件可加载性冒烟测试（容器内）
docker exec -w /MaiMBot maim-bot-core .venv/bin/python -c "
import importlib.util, sys
from pathlib import Path
p = Path('/MaiMBot/plugins/opencode-session-header/plugin.py')
spec = importlib.util.spec_from_file_location('probe', str(p), submodule_search_locations=[str(p.parent)])
m = importlib.util.module_from_spec(spec); sys.modules['probe'] = m; spec.loader.exec_module(m)
i = m.create_plugin()
print([c.get('name') for c in i.get_components()])
print([(x['client_type'], x['metadata']['handler_name']) for x in i.get_llm_providers()])"
```

文件结构：

```
opencode-session-header/
├── _manifest.json        插件清单（严格模式）
├── plugin.py             插件类：LLM Provider、命令、工具、配置模型
├── session_identity.py   端点识别、会话种子提取、SHA-256 截断哈希、种子持久化
├── openai_wire.py        Context Item ↔ OpenAI 线格式（纯函数）
├── openai_transport.py   HTTP 传输与鉴权
├── tests/                离线单元测试
├── README.md
├── LICENSE               MIT
└── .gitignore
```

## 11. 发布信息

| 项 | 值 |
|---|---|
| 插件 ID | `maibot.opencode-session-header` |
| 版本 | `1.1.1` |
| 作者 | deepseekv41flash |
| 仓库 | https://github.com/deepseekv41flash/maibot-opencode-session-header |
| 许可证 | MIT |

插件市场按 **Release 版本**收录，Tag 与 `_manifest.json` 的 `version` 必须一致
（`v1.1.1` ↔ `1.1.1`）。`host_application` / `sdk` 的上界按官方建议置于
`999.999.999`，只认真约束 `min_version`。

## 12. 变更记录

### 1.1.1

* 文档：第 5 节与第 7 节补充请求转发面说明（请求体与鉴权头会发往 `base_url` 及
  `endpoint.extra_base_url_prefixes` 命中的地址）、`session_id_env` 不应填密钥类变量名、
  `verbose` 日志会输出令牌本身；
* 文档：修正 `[plugin].config_version` 的默认值标注（`1.1.0`）；
* 版本号同步至 `1.1.1`（`_manifest.json` 与 `@LLMProvider` 的 `version`）。

**行为无变化**：1.1.1 不改变任何代码逻辑，配置结构未变，`config_version` 保持 `1.1.0`。

### 1.1.0

* 首个公开版本：命中 `opencode.ai` 端点时注入稳定的 `X-OpenCode-Session`；
* 支持 `chat_completions` 与 `responses` 两种线上协议，并新增 `[wire]` 配置节。

## 13. 许可

MIT，见 [LICENSE](LICENSE)。
