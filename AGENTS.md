# 项目说明

这些说明适用于在本工作区工作的每一次 Codex 对话和代理，包括已授权委派时的子代理。

## 云端仓库与同步

- 本项目的云端仓库是 [GitHub 上的 eason8319/phd-research](https://github.com/eason8319/phd-research)。
- 用户要求将项目内容与云端同步时，查阅本节，并以该仓库为同步目标。
- 同步前检查本地 Git 设置、远程配置和当前变更，以确定所请求同步需要的步骤。

## Cursor 规则兼容

- 开始项目工作前，阅读本工作区 `.cursor/rules/` 目录中的当前规则。
- 始终遵守标记为 `alwaysApply: true` 的规则。其他规则按其文件模式、声明的范围和当前任务适用。
- Cursor 规则文件是项目约定的权威来源。规则变更后重新阅读相关规则；不要依赖其他对话中记住的副本。
- 委派已授权的工作时，传递工作区根目录和适用规则，并要求代理在行动前阅读本文件和相关 Cursor 规则。
- 这些项目规则仍服从更高优先级的系统说明、开发者说明以及用户的明确指示。

## 实验管理

- 研究目录、共用代码、阶段执行代码、结果回传和审核后报告，始终遵守 [`.cursor/rules/experiment-management.mdc`](.cursor/rules/experiment-management.mdc)。
- 共用代码放在 `research/src/kvstudy/`。各阶段的执行代码和唯一指导文档放在 `research/phaseN/`。当前只有第一阶段。
- 代码、指导文档和审核后报告的权威维护位置在本地。正式结果整理后回传到本地 `research/phaseN/results/`，同一实验的新结果覆盖旧结果。运行过程不写报告；审核通过后才作图并写报告。根目录 `models/` 与 `research/.runtime/` 仅在服务器，不进入 Git。一次性测试及其结果不留在项目中。
- 项目内代码与注释遵守 [`.cursor/rules/experiment-code.mdc`](.cursor/rules/experiment-code.mdc)。项目内报告遵守 [`.cursor/rules/experiment-reports.mdc`](.cursor/rules/experiment-reports.mdc)。

## 文献综述

每月收集论文和预印本时使用 `docs/literature-survey-monthly.md`（检索约束与月度记录），并同步 `docs/literature-survey-reading-list.md` 与 `docs/literature-survey.bib`。新增、剔除、发表状态变化和重新编号都必须同时更新这两个文件；完成前校验引用键一一对应、书目元数据以及已有引用。规则：`.cursor/rules/literature-survey.mdc`。触发语：`做本月文献更新`。不要在本仓库添加检索脚本。

## 工作区隔离

以下要求与 `.cursor/rules/workspace-isolation.mdc` 一致，以便在 Codex 的项目说明中直接可见。

只允许读取、搜索或编辑以下两棵项目目录：

- 本地：本仓库当前打开的 Cursor 工作区根目录（`f:\phd-research`）。
- 服务器：SSH 主机 `myserver` 上的 `/cluster/home/zengy/phd-research`。

这两棵根目录之内的目录均在范围内。SSH 与 shell 命令必须从对应的根目录开始，不得列出、读取或搜索任何上级目录或其他路径。

对本项目在 `myserver` 上打开的每条 SSH 命令或会话：

- 远程命令在任何检查、脚本或任务之前，必须以 `cd /cluster/home/zengy/phd-research || exit 1` 开头。进入该目录失败则立即停止。
- 每次新建连接都执行这一进入步骤；上一条 SSH 命令的工作目录不会延续到下一次连接。
- 交互式会话中，开始项目工作前先进入该目录。此后的全部操作都留在项目树内，并遵守下列隔离规则。

- 在当前 Windows 机器上，运行 `docs/server-access.md` 中的 `ssh` 命令。不要为该连接新增项目脚本。不要通过检索其他对话来查找连接设置。
- 仅用于认证的例外：OpenSSH 可以仅为连接本项目服务器而使用 `C:\Users\User\.ssh\config`、`C:\Users\User\.ssh\id_ecdsa_zengy` 及其已有的主机密钥记录。不得读取、显示、复制或提交私钥内容，也不得浏览其他 SSH 文件。工具层面的权限检查仍然适用。

- 不要打开其他本地项目、服务器上的同级目录、其他用户的文件或 SSH 主目录。
- 这两棵根目录之外，仅在需要时可以使用本工作区自身的 Cursor 项目元数据，以及上述仅用于认证的 SSH 例外。
- 不要搜索、读取或引用过往对话、代理对话记录、聊天记录或缓存的云端聊天。
- 不要把用户的个人代理存储、全局记忆或其他聊天当作背景资料。若某一事实不在这两棵根目录或当前线程中，应向用户询问，或在相应根目录内查找。
- 不要把此前在其他项目中查看过的文件当作相关上下文。
- 若信息似乎只存在于旧聊天中，应如实说明，并基于这两棵根目录开展工作，不要去检索那次聊天。
