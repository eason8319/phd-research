# 本项目的服务器连接

本说明供本项目的其他对话直接使用，无需读取任何历史聊天。适用于当前 Windows 电脑及其已有的 SSH 凭据。

## 已验证的连接信息

- 本地项目：`F:\phd-research`。
- SSH 主机别名：`myserver`；登录用户名：`zengy`。`zengy` 不是主机别名。
- 服务器地址：`202.120.39.87`，端口 `22`。
- SSH 配置：`C:\Users\User\.ssh\config`。
- SSH 身份文件：`C:\Users\User\.ssh\id_ecdsa_zengy`。
- 服务器项目：`/cluster/home/zengy/phd-research`。

配置和身份文件仅由 SSH 客户端用于认证，不要读取、显示、复制或提交私钥内容。身份文件保留在原位置，不放入项目。

## 连接命令

在本地项目根目录 `F:\phd-research` 执行以下命令。不要在项目中添加连接脚本。连接成功后应输出 `/cluster/home/zengy/phd-research`：

```powershell
ssh -F C:\Users\User\.ssh\config -i C:\Users\User\.ssh\id_ecdsa_zengy -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -o ConnectionAttempts=1 -o ServerAliveInterval=30 -o StrictHostKeyChecking=yes myserver "cd /cluster/home/zengy/phd-research || exit 1; pwd -P"
```

执行具体的项目命令时，替换最后的 `pwd -P`。例如查看主机名：

```powershell
ssh -F C:\Users\User\.ssh\config -i C:\Users\User\.ssh\id_ecdsa_zengy -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -o ConnectionAttempts=1 -o ServerAliveInterval=30 -o StrictHostKeyChecking=yes myserver "cd /cluster/home/zengy/phd-research || exit 1; hostname"
```

每次调用都是新的连接。远程命令的第一条必须是 `cd /cluster/home/zengy/phd-research || exit 1`。进入失败立即停止，SSH 的失败状态会返回调用方。

这条命令只设置初始目录，不提供操作系统级隔离；后续命令仍须遵守 `AGENTS.md` 和 `.cursor/rules/workspace-isolation.mdc`，不得访问其他项目、父目录或聊天记录。

## 连接失败时

- **无法识别主机别名**：使用上述命令，确保传入显式的 SSH 配置路径；不要执行 `ssh zengy`。
- **访问 SSH 配置或身份文件被拒绝**：本项目允许 SSH 客户端将这些文件用于本服务器认证，但每个对话的工具沙箱仍可能需要单独授权。在支持权限申请的执行工具中申请本次连接所需权限；不要通过复制密钥到项目、放宽文件权限或关闭隔离来处理。
- **Permission denied (publickey)**：认证未通过，记录 SSH 错误，核对配置是否仍对应上述账号和身份文件；不要打印私钥。
- **Host key verification failed**：停止并核对服务器身份，不关闭主机密钥校验。
- **连接超时**：检查当前电脑是否能访问该服务器地址及所需网络；不要将超时误判为项目目录问题。
- **项目目录不存在或不能进入**：停止，不能退回 SSH 家目录继续执行。

2026-09-23 在本机复测：普通工具沙箱报告配置和身份文件 `Permission denied`；申请执行权限后，成功返回 `/cluster/home/zengy/phd-research`。

使用支持 `exec_command` 的对话时，将工作目录设为 `F:\phd-research`，执行上述 `ssh` 命令。若普通沙箱出现上述权限错误，使用执行工具提供的 `sandbox_permissions: "require_escalated"` 为同一命令申请授权。该申请仍由平台审批；若被拒绝，应报告具体原因，不得换一种方式绕过拒绝。

其他对话仍可能受到各自的网络权限、沙箱和运行位置限制。项目文件共享连接方法，不共享一次对话中的审批状态，也不向另一台电脑提供凭据。
