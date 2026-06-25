# Render 部署指南（Docker + ttyd + noVNC）

本指南使用 **Docker + 网页终端 + noVNC** 部署，**不修改 `app.py` 的任何逻辑**。你在网页里看到的终端，等价于在本机运行 `python app.py`；需要 UW 登录时，通过 noVNC 在虚拟浏览器里操作。

---

## 架构说明

| 组件 | 作用 |
|------|------|
| **ttyd** | 在浏览器里提供终端，运行 `python app.py` |
| **Xvfb** | 虚拟显示器，让 `headless=False` 的 Playwright 能跑 |
| **x11vnc + noVNC** | 把虚拟显示器暴露到 `/vnc/`，用于 UW 登录 |
| **nginx** | 统一入口，转发到 ttyd 和 noVNC |
| **Render** | 托管 Docker 容器，提供公网 URL |

访问地址：

- **终端（聊天）**：`https://你的服务名.onrender.com/`
- **虚拟浏览器（UW 登录）**：`https://你的服务名.onrender.com/vnc/vnc.html?autoconnect=true&resize=scale`
- **健康检查**：`https://你的服务名.onrender.com/health`

---

## 前置要求

1. **GitHub 账号**（Render 从 GitHub 拉代码部署）
2. **Render 账号**：https://render.com
3. **OpenAI API Key**（环境变量 `OPENAI_API_KEY`）
4. **课程数据** `courses.json`（构建镜像时必须存在）
5. **推荐 Render 套餐**：Starter（$7/月）或更高；免费版 512MB 内存很可能不够（Chroma + LangChain + Playwright + Xvfb）

---

## 第一步：准备本地项目

### 1.1 确认 `courses.json` 存在

`courses.json` 默认在 `.gitignore` 里，但 **部署必须带上它**。

在项目根目录确认文件存在：

```powershell
dir courses.json
```

若还没有，先按 README 准备课程数据。

### 1.2 将 `courses.json` 加入 Git（仅用于部署）

因为 Render 从 GitHub 构建，需要把 `courses.json` 推送到仓库（文件可能较大，若超 GitHub 限制需用 Git LFS 或 Persistent Disk，见文末）。

```powershell
cd C:\Users\1\Desktop\uw-course-assistant-agent
git add -f courses.json
git add Dockerfile requirements.txt docker/ .dockerignore DEPLOY.md
git status
git commit -m "Add Docker deployment for Render (ttyd + noVNC)"
git push origin main
```

若主分支名是 `master`，把 `main` 改成 `master`。

### 1.3（可选）本地预构建向量库，加快首次启动

首次部署会在容器里自动跑 `build_vector_db.py`（约 5–15 分钟，需消耗 OpenAI Embedding 费用）。

若想跳过在线构建，可在本地先构建，并把结果打进镜像：

```powershell
# 本地需已配置 OPENAI_API_KEY（.env 或环境变量）
python build_vector_db.py
```

然后临时从 `.dockerignore` 中去掉 `uw_chroma_db/`，并在 `Dockerfile` 里增加：

```dockerfile
COPY uw_chroma_db/ ./uw_chroma_db/
```

再提交推送。否则首次打开终端时会先看到向量库构建进度，属正常现象。

---

## 第二步：本地验证 Docker（推荐）

安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/) 后：

```powershell
cd C:\Users\1\Desktop\uw-course-assistant-agent

docker build -t uw-course-agent .

docker run --rm -p 10000:10000 -e OPENAI_API_KEY=sk-你的key -e PORT=10000 uw-course-agent
```

浏览器打开：

- 终端：http://localhost:10000/
- noVNC：http://localhost:10000/vnc/vnc.html?autoconnect=true&resize=scale

在终端里输入问题测试；若触发课表查询且需 UW 登录，打开 noVNC 完成登录。

确认无误后再部署到 Render。

---

## 第三步：在 Render 创建 Web Service

### 3.1 新建服务

1. 登录 https://dashboard.render.com
2. 点击 **New +** → **Web Service**
3. 连接 GitHub，选择仓库 `uw-course-assistant-agent`
4. 若首次使用，授权 Render 访问你的 GitHub

### 3.2 服务配置（逐项填写）

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **Name** | `uw-course-agent`（自定） | 决定 URL：`https://uw-course-agent.onrender.com` |
| **Region** | 选离你近的 | |
| **Branch** | `main` | 与仓库主分支一致 |
| **Root Directory** | 留空 | |
| **Runtime** | **Docker** | 必须选 Docker，不要选 Python |
| **Instance Type** | **Starter** 或更高 | 强烈建议不用 Free |
| **Auto-Deploy** | On | push 后自动重新部署 |

**Docker 相关**（选 Docker 后一般自动识别）：

| 配置项 | 值 |
|--------|-----|
| Dockerfile Path | `./Dockerfile` |
| Docker Command | 留空（使用 Dockerfile 的 `CMD`） |

**不要填** Build Command / Start Command（Docker 模式不需要）。

### 3.3 环境变量

在 **Environment** → **Environment Variables** 添加：

| Key | Value |
|-----|--------|
| `OPENAI_API_KEY` | `sk-...` 你的 OpenAI Key |

可选（一般不用改）：

| Key | Value |
|-----|--------|
| `PORT` | Render 会自动注入，无需手动设置 |

### 3.4 健康检查（重要）

在 **Settings** → **Health Check**：

| 配置项 | 值 |
|--------|-----|
| **Health Check Path** | `/health` |

不要用默认的 `/`。部署时 ttyd 可能仍在构建向量库，但 `/health` 会立即返回 `ok`，避免部署失败。

### 3.5（推荐）Persistent Disk 持久化向量库

避免每次重新部署都重建 Chroma：

1. **Settings** → **Disks** → **Add Disk**
2. **Mount Path**：`/app/uw_chroma_db`
3. **Size**：1 GB（可按数据量调整）

首次部署仍会在磁盘上构建一次；之后 redeploy 会保留向量库。

### 3.6 创建并等待部署

点击 **Create Web Service**。Render 会：

1. 拉取 GitHub 代码
2. `docker build`（安装依赖、复制 `courses.json`）
3. 启动容器（Xvfb → VNC → ttyd → nginx）
4. 健康检查 `/health` 通过后显示 **Live**

首次若未预构建向量库，打开终端后需等待 `build_vector_db.py` 完成，才会出现 `✅ Agent is ready!`。

---

## 第四步：使用网站

### 4.1 打开终端聊天

访问：`https://你的服务名.onrender.com/`

与本地一样：

```
🙋 You: 有哪些入门级的 CSE 课程？
```

输入 `quit` 或 `exit` 退出（在网页终端里会结束当前会话；刷新页面可重新进入，但对话历史会重置）。

### 4.2 UW 课表登录（noVNC）

当问题触发 `get_time_schedule` 且需要 UW NetID 时，终端会显示：

```
⚠️ Login required. Please complete UW NetID login and 2FA in the popped-up browser window...
⏳ Waiting for you to log in...
```

此时：

1. **新开标签页** 打开：  
   `https://你的服务名.onrender.com/vnc/vnc.html?autoconnect=true&resize=scale`
2. 在 noVNC 画面里完成 UW 登录和 2FA
3. 登录成功后，`app.py` 里原有的 `wait_for_url` 会继续执行，Agent 返回课表结果

这与本机「终端 + 弹出浏览器」的流程一致，**无需改 `app.py`**。

### 4.3 多人使用说明

- **任何人** 都能打开 URL（默认无密码）
- **同时多人** 会共用同一个终端和对话历史，输入会混在一起
- 适合个人或小团队错开使用；不适合作为公开多用户产品

---

## 第五步：更新与重新部署

修改代码或数据后：

```powershell
git add .
git commit -m "Update ..."
git push
```

Render 会自动重新构建并部署。若挂载了 Persistent Disk，`uw_chroma_db` 一般保留；若更新了 `courses.json` 且需重建库，在终端里手动删库或清空磁盘目录后重启服务。

---

## 安全建议（强烈建议生产环境配置）

当前方案 **终端和 noVNC 对公网开放**，风险较高。建议至少做一种：

1. **Render 不公开 URL**：仅自己保存链接，不传播（最弱）
2. **在 Render 前加 Cloudflare Access / 简单认证**（需自定义域名）
3. **nginx Basic Auth**：在 `docker/start.sh` 的 nginx 配置里增加 `auth_basic`（需自行改配置并设置用户名密码环境变量）

切勿把 `OPENAI_API_KEY` 提交到 Git；只用 Render 环境变量。

---

## 常见问题

### 部署失败：`courses.json not found`

- 本地构建：确保根目录有 `courses.json`
- Render：确保已 `git add -f courses.json` 并 push

### 部署失败：内存不足 / OOM

- 升级到 Starter 或更高内存套餐
- 或本地预构建 `uw_chroma_db` 打进镜像，减少运行时内存峰值

### 终端一直卡在 “building vector database”

- 正常，首次需调用 OpenAI Embedding，等待 5–15 分钟
- 检查 `OPENAI_API_KEY` 是否正确、账户是否有余额

### noVNC 黑屏

- 确认 Xvfb 已启动（看 Render **Logs**）
- 先访问终端触发一次课表查询，再在 noVNC 里看是否出现浏览器窗口
- 尝试刷新 noVNC 页面

### Playwright / Chromium 报错

- 必须使用 **Docker** 部署，不要用 Render 原生 Python
- 镜像基于 `mcr.microsoft.com/playwright/python`，已包含 Chromium

### 健康检查失败

- 确认 Health Check Path 为 `/health`，不是 `/`

### 闲置后很慢

- 免费/低档套餐闲置会休眠，首次访问需冷启动 30 秒～1 分钟

### `courses.json` 太大无法推 GitHub

- 使用 [Git LFS](https://git-lfs.github.com/)
- 或首次部署后通过 Render Shell 上传（若套餐支持）
- 或把 `courses.json` 放在对象存储，在 `docker/run_agent.sh` 开头用 `curl` 下载（会改动启动脚本，但不改 `app.py`）

---

## 项目内相关文件

| 文件 | 说明 |
|------|------|
| `Dockerfile` | 镜像定义（Playwright + ttyd + noVNC + nginx） |
| `docker/start.sh` | 容器启动：Xvfb、VNC、ttyd、nginx |
| `docker/run_agent.sh` | ttyd 入口：按需构建向量库 → `python app.py` |
| `requirements.txt` | Python 依赖 |
| `.dockerignore` | 构建时排除 venv、`.env` 等 |
| `app.py` | **未修改**，保持原有 Agent 逻辑 |

---

## 快速检查清单

- [ ] `courses.json` 已存在并已 push 到 GitHub
- [ ] Render Runtime 选 **Docker**
- [ ] 环境变量 `OPENAI_API_KEY` 已设置
- [ ] Health Check Path = `/health`
- [ ] Instance Type ≥ Starter（推荐）
- [ ] （可选）Persistent Disk 挂载到 `/app/uw_chroma_db`
- [ ] 本地 `docker build` + `docker run` 测试通过
- [ ] 终端可聊天，noVNC 可完成 UW 登录

完成以上步骤后，即可在公网以网站形式使用完整功能的 UW Course Agent。
