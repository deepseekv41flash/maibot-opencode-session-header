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

1. 注册**两个** Provider：`opencode-session`（chat_completions）与
   `opencode-session-responses`（responses），各自接管指向自己的模型请求；
2. 读取本次请求的 `api_provider.base_url`，判断是否命中 `opencode.ai`（含子域）；
3. 命中时按优先级解析会话种子，做 SHA-256 截断哈希，注入会话头；
4. 未命中时**不添加任何请求头**，请求体原样转发 —— 与内置客户端行为一致；
5. 把网关响应还原成宿主可消费的结构（正文 / 推理内容 / 工具调用 / usage）。

线上协议由你选用的 `client_type` 决定，**不再有全局协议开关**，因此两种协议
可以同时使用（见 5.1）。

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

插件提供两种 `client_type`，**必须**把它们之一填到 opencode 服务商上，插件才会生效：

| `client_type` | 走哪个端点 |
|---|---|
| `opencode-session` | `POST {base_url}/chat/completions` |
| `opencode-session-responses` | `POST {base_url}/responses` |

**WebUI（推荐）**：设置 → 模型 → 编辑 opencode 的 API 服务商 →
「客户端类型」选择上表中的一个。

**手改 `config/model_config.toml`**：

```toml
[[api_providers]]
name = "opencode"
base_url = "https://opencode.ai/zen/v1"
api_key = "sk-..."          # 你自己的密钥
client_type = "opencode-session"   # ← 关键：默认是 "openai"
```

保存后重启或热重载模型配置即可。想让不同模型分别使用两种协议，见 5.1。

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
| `verbose` | `false` | 记录每次注入的端点、协议、是否命中、注入的**头名**与种子来源（**不含令牌值**）。日志会暴露部署细节，排查完建议关掉 |

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
| `responses_store` | `false` | 仅 `opencode-session-responses` 使用：是否让网关保存响应（`store`） |

> 1.2.0 起移除了 `wire_protocol`：协议改由 `client_type` 决定，不再有全局开关。

## 5.1 两个客户端类型（协议由 client_type 决定）

插件注册了**两个**独立的 LLM Provider。协议不再由全局开关切换，而由你在模型配置里
选用的 `client_type` 决定：

| `client_type` | 端点 | 备注 |
|---|---|---|
| `opencode-session` | `POST {base_url}/chat/completions` | 历史名称，向后兼容 |
| `opencode-session-responses` | `POST {base_url}/responses` | Responses API |

两者**可以同时启用**：同一台机器人上，一部分模型走 chat、另一部分走 responses。

### 怎么让两种协议并存

`client_type` 挂在 **API Provider** 上，而模型通过 `api_provider` 引用服务商。
所以要并存，需要建**两个 API Provider**（`base_url` / `api_key` 相同，只有 `name`
与 `client_type` 不同）：

```toml
[[api_providers]]
name = "opencode"                  # 给 chat 协议用
base_url = "https://opencode.ai/zen/go/v1"
api_key = "sk-..."
client_type = "opencode-session"

[[api_providers]]
name = "opencode-responses"        # 给 responses 协议用
base_url = "https://opencode.ai/zen/go/v1"
api_key = "sk-..."
client_type = "opencode-session-responses"
```

再让不同模型分别引用它们：

```toml
[[models]]
model_identifier = "deepseek-v4.1-flash"
name = "deepseek-v4.1-flash"
api_provider = "opencode"                   # → /chat/completions

[[models]]
model_identifier = "some-responses-model"
name = "some-responses-model"
api_provider = "opencode-responses"         # → /responses
```

WebUI 同理：复制一份 API 服务商，改个名字并选对应的客户端类型。

### 令牌行为

令牌种子按 `服务商名|主机` 生成。上面两个服务商名字不同，因此各自持有一枚**独立的稳定令牌**，
互不影响。两个 `client_type` 的端点识别、种子解析、哈希注入逻辑完全一致。

### 选哪个

实测 `opencode.ai/zen/go/v1` 两个端点都存在：

```
POST /zen/go/v1/chat/completions -> 401 AuthError: Missing API key.
POST /zen/go/v1/responses        -> 401 AuthError: Missing API key.
```

（无鉴权探测；返回 AuthError 而非 404，说明路由与模型都被接受。）

**默认建议 `opencode-session`（chat）**，它与内置客户端行为最接近。
只有当中转网关只实现 `/responses`、或你确实需要 Responses 的结构化输出（`text.format`）时，
才给对应模型单独挂 `opencode-session-responses`。

### responses 模式与内置客户端的差异

主程序内置的 responses 客户端依赖 **provider replay fragment** 把推理项原样回传
（`_can_replay_item()` 会校验 client_type、端点指纹与模型名）。插件这条路拿不到 replay ——
`serialize_context_item_snapshot()` 明确不把 replay 放进快照，宿主反序列化时也不会还原。
因此插件在 responses 模式下**直接省略 reasoning 项**，只发 `function_call` +
`function_call_output`。

实际影响比想象中小：

* 单轮问答、普通工具调用：**没有区别**；
* 多轮工具循环：丢掉的是**跨轮次的推理连续性**（质量层面），不是报错。
  已知会真报错的是另外两种写法 —— 回传 `rs_...` ID（`store=false` 下报
  `Item with id ... not found`）或回传残缺的 reasoning 项（报
  `reasoning was provided without its required following item`）。插件两种都不做，
  所以**不会因为这一项被网关拒绝**；
* 主程序内置客户端在 replay 作用域不匹配时，走的也是同样的「丢弃 reasoning」降级路径。

## 6. 命令与工具

| 类型 | 名称 | 用途 |
|---|---|---|
| 命令 | `/opencode_session`（别名 `/oc_session`） | 查看注入开关、匹配域名、两个客户端类型、令牌预览 |
| 命令 | `/opencode_session_reset`（别名 `/oc_session_reset`） | 清空兜底种子，下次请求重新生成 |
| 工具 | `opencode_session_token` | 让模型计算某个 Base URL / 客户端类型会注入什么头，用于排障 |

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
* `verbose` 日志会输出端点、协议、是否命中、注入的**头名**与种子来源（如 `fallback:provider|host`、
  `env:NAME`）——**不输出令牌值**，也不含聊天内容；但种子来源会暴露部署细节，请按你所在部署的
  日志留存策略处理；
* **重定向**：HTTP 传输沿用 httpx 的默认行为（`follow_redirects=True`）。当重定向目标与
  原地址**不同源**（协议 / 主机 / 端口任一不同）时，httpx 会剥离标准 `Authorization` 头
  （同主机 `http → https` 升级除外），但**不会剥离自定义鉴权头**——即 `auth_type = "header"`
  时配置的 `auth_header_name` 会被带到重定向目标；`307` / `308` 还会把请求体重发到新主机。
  因此请让自建 / 中转网关直接返回结果，不要做跨域重定向；
* 插件只读取配置里的 URL 与请求头，不读取、不上报任何聊天内容。

## 8. 排障

**仍然 400 MissingSessionID**
1. 确认该 API 服务商的 `client_type` 是 `opencode-session` 或 `opencode-session-responses`
   （见第 4 节）；`openai` 等内置类型不会触发本插件；
2. 打开 `[session_header].verbose`，看日志里是否出现
   `OpenCode 会话头注入: ... protocol=... matched=True`；
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
  `input`）；遇到未知 Item 类型会跳过而不是让请求失败，同时按数量与类型记一条 warning，
  避免上下文静默缺块；
* `opencode-session-responses`（responses 协议）下拿不到 provider replay fragment，
  插件会省略 reasoning 项，影响是跨工具轮次的推理连续性（质量层面），不会因此被网关拒绝，
  详见第 5.1 节。没有特殊需求请让模型继续使用 `opencode-session`（chat）。

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
| 版本 | `1.1.3` |
| 作者 | deepseekv41flash |
| 仓库 | https://github.com/deepseekv41flash/maibot-opencode-session-header |
| 许可证 | MIT |

插件市场按 **Release 版本**收录，Tag 与 `_manifest.json` 的 `version` 必须一致
（`v1.1.3` ↔ `1.1.3`）。`host_application` / `sdk` 的上界按官方建议置于
`999.999.999`，只认真约束 `min_version`。

## 12. 变更记录

### 1.2.0

* **新功能**：把线上协议拆成两个独立的 `client_type` ——
  `opencode-session` 走 `POST /chat/completions`，`opencode-session-responses` 走
  `POST /responses`。两者可同时注册，因此同一台机器人上不同模型可以分别使用两种协议；
* **配置变更**：移除 `[wire].wire_protocol`（含 `auto` 自动回退）。协议改由 `client_type`
  决定，不再需要全局开关；`[wire]` 只剩 `responses_store`；
* 两个 client_type 共用同一套端点识别、种子解析与哈希注入逻辑；令牌种子仍按
  `服务商名|主机` 生成，因此并存的两个服务商各持一枚独立稳定令牌；
* `opencode_session_token` 工具新增可选参数 `client_type`，并回报两个 client_type 的映射；
* `config_version` 提升至 `1.2.0`，旧配置会自动迁移（补 `responses_store`、移除
  `wire_protocol`）。

**迁移注意**：如果你的 `wire_protocol` 原来是 `responses`（或 `auto` 且已回退到 responses），
升级后该服务商默认回到 chat_completions。要保留 responses，请再建一个 API Provider，
把它的 `client_type` 设为 `opencode-session-responses`（见 5.1）。

### 1.1.3

* 修复：`endpoint.extra_base_url_prefixes` 改为按**路径段边界**匹配——此前前缀
  `https://gw.example.com/opencode` 会连带命中 `https://gw.example.com/opencode-evil.example`，
  把会话令牌注入到非预期地址；
* 修复：`opencode_session_token` 工具改为**只读**路径（新增 `SessionSeedStore.peek` / `peek_host`），
  不再为模型试探过的每个域名创建条目，避免 `session_seeds.json` 无上限增长；无既有种子时给出
  不落盘的只读预览值并在结果中注明；
* 改进：线格式投影遇到无法识别的 Context Item 时，按数量与类型记一条 warning（此前静默丢弃）；
* 文档纠错：`verbose` 日志实际输出端点、协议、命中情况、注入的**头名**与种子来源，**不含令牌值**，
  1.1.1 起的措辞过宽，本版改正；
* 版本号同步至 `1.1.3`（`_manifest.json` 与 `@LLMProvider` 的 `version`）。

**兼容性**：前缀匹配收紧属**行为变更**——若此前依赖「`/opencode` 也能命中 `/opencode-xxx`」这类
宽松匹配，升级后需把前缀写成完整路径段（如 `https://gw.example.com/opencode/v1`）。配置结构与其余
逻辑未变，`config_version` 保持 `1.1.0`。

### 1.1.2

* 变更：`opencode_session_token` 工具返回体移除 `seed_source` 字段，不再把部署细节
  暴露给模型（`verbose` 日志中仍保留，便于排障）；
* 文档：第 7 节补充重定向行为说明——跨域重定向时 httpx 会剥离标准 `Authorization` 头，
  但不会剥离自定义鉴权头，且 `307` / `308` 会把请求体重发到新主机；
* 版本号同步至 `1.1.2`（`_manifest.json` 与 `@LLMProvider` 的 `version`）。

**兼容性**：工具返回体字段减少属向后兼容变更；HTTP 传输层与配置结构未改，
`config_version` 保持 `1.1.0`。

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
