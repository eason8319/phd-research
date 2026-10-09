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
ssh -F "C:\Users\User\.ssh\config" -i "C:\Users\User\.ssh\id_ecdsa_zengy" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -o ConnectionAttempts=1 -o ServerAliveInterval=30 -o StrictHostKeyChecking=yes myserver "cd /cluster/home/zengy/phd-research || exit 1; pwd -P"
```

执行具体的项目命令时，替换最后的 `pwd -P`。例如查看主机名：

```powershell
ssh -F "C:\Users\User\.ssh\config" -i "C:\Users\User\.ssh\id_ecdsa_zengy" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -o ConnectionAttempts=1 -o ServerAliveInterval=30 -o StrictHostKeyChecking=yes myserver "cd /cluster/home/zengy/phd-research || exit 1; hostname"
```

每次调用都是新的连接。远程命令的第一条必须是 `cd /cluster/home/zengy/phd-research || exit 1`。进入失败立即停止，SSH 的失败状态会返回调用方。

这条命令只设置初始目录，不提供操作系统级隔离；后续命令仍须遵守 `AGENTS.md` 和 `.cursor/rules/workspace-isolation.mdc`，不得访问其他项目、父目录或聊天记录。

## 计算节点执行方式

按用户 2026-10-09 的要求，实验、实现核验和 GPU 资源检查使用交互式计算节点会话，并显式选择满足容量要求的节点，不设置至少 40 GiB 空闲的统一条件。node8 为 48GB，其余节点为 24GB。每次工作前，按实际模型、输入长度、输出预算、缓存 dtype 与副本、临时张量、logits 保存及已有实测峰值估算显存，并记录不确定性余量；24GB 足够时可以使用其余节点，需要更大容量时使用 node8。进入计算节点后检查实际设备、总容量、实时空闲显存和设备可用性，目标 GPU 满足该次估算才启动。显存不足或设备异常时保留记录并换空闲 GPU、节点或等待，不缩短冻结实验。仍不提交 `sbatch` 批处理作业。统一命令如下，将 `n` 替换为实际节点号，例如 `8`：

```bash
srun -p gpu -c 32 -n 1 -N 1 --nodelist=node[n] --pty /bin/bash
```

从 Windows 连接时，在上面的 `ssh` 之后、`-F` 之前加 `-tt`，并使用支持终端交互的会话。远程命令仍先执行 `cd /cluster/home/zengy/phd-research || exit 1`，然后执行上述 `srun`。进入计算节点 shell 后，在任何检查或实验之前再次执行 `cd /cluster/home/zengy/phd-research || exit 1`；全部后续操作留在项目树内。

进入节点后核对实际 GPU、实时空闲显存与该次峰值估算，使用项目内冻结运行环境。更换设备时核对 BF16 和软件兼容性，保留实际设备与逐请求峰值。实验继续保持每次一个模型、一个进程，并保留逐题恢复进度。会话结束后按实际状态记录中断或完成，不将排队、获得 shell 或启动进程记为实验完成。

长时间等待显存或运行实验时，在服务器端的持久交互终端中执行上述 `srun`，以便本地 SSH 连接结束后继续运行。使用 `tmux` 时，将其显式配置文件和套接字放在本项目的 `research/.runtime/` 内；计算节点 shell 仍须先进入项目根目录。启动后通过另一条连接确认终端存活、等待日志持续更新，再记为自动等待或执行。

当前补测的持久终端名为 `phase1-completion-node8`，套接字为 `research/.runtime/phase1-completion.tmux.sock`。按连接命令进入项目根目录后，可用以下命令接回该终端：

```bash
tmux -S /cluster/home/zengy/phd-research/research/.runtime/phase1-completion.tmux.sock attach -t phase1-completion-node8
```

接回终端后，`Ctrl+b` 再按 `d` 可脱离终端并保持服务器进程运行。不要在等待或实验前台运行时退出计算节点 shell；重启前先确认当前交互分配与进程是否仍存活，避免重复运行。

## Codex 项目级长期授权

本项目的执行允许规则保存在 [`.codex/rules/ssh-myserver.rules`](../.codex/rules/ssh-myserver.rules)，分别匹配上述普通连接，以及在 `ssh` 后加 `-tt` 的交互式连接。规则要求认证文件路径、全部固定选项的顺序和值、主机 `myserver` 一致，允许替换最后的远程项目命令。其他主机、身份文件或不匹配的调用不会获得这份允许规则的授权。

该文件是 Codex 的命令审批配置，与 `.cursor/rules/` 中的项目行为约定不同。按 [Codex 官方说明](https://learn.chatgpt.com/docs/agent-configuration/rules)，项目 `.codex/` 层必须受信任，新增或修改规则后须重启 Codex 才会加载。更严格的平台或管理员规则仍可要求审批或拒绝执行。当前会话中的“允许一次”不代表保存了长期授权，也不能据此判断新规则已经加载。

规则允许匹配的 SSH 命令在本地沙箱外执行，不提供服务器目录的操作系统级隔离。每次连接仍须先进入服务器项目根目录，所有后续操作遵守 `AGENTS.md` 与 `.cursor/rules/workspace-isolation.mdc`。认证文件继续保留在原位置，由 SSH 客户端使用。

规则语法与匹配范围可在本地项目根目录用 `codex execpolicy check --rules .codex/rules/ssh-myserver.rules -- <完整 SSH 命令的参数>` 校验；该检查不连接服务器。校验通过不等于当前应用已经加载规则。重启后再执行本页的目录确认命令，结合实际弹窗确认是否实现免重复审批。

2026-10-05 使用本机 `codex-cli 0.159.0-alpha.12.1` 校验：普通连接和交互式连接均匹配 `allow`；其他主机、其他身份文件及将 `StrictHostKeyChecking` 改为 `no` 的调用均不匹配。新规则写入后的未重启会话仍在默认沙箱下拒绝认证文件；申请本次执行权限后成功返回 `/cluster/home/zengy/phd-research`。用户重启 Codex 后，系统已提供普通及交互式连接的已保存授权前缀；再次执行本页的普通连接命令，使用默认执行参数即成功返回项目目录，未申请额外执行权限。普通连接的重启后直接执行已验证；交互式连接完成规则匹配校验，尚未单独进行实际连接测试。

### SSH 数据传输与自动审批

传输调用同样应保持上述固定 SSH 参数与顺序，直接执行已授权的 `ssh` 命令。通过执行工具的标准输入、标准输出收发数据时，须区分启动命令与向已有终端发送输入两个步骤：现有允许规则匹配 SSH 启动命令，不能据此保证后续 `write_stdin` 等终端输入操作也免审批。若将 SSH 包在其他程序或复杂本地脚本中，或改用 `scp`、`sftp`，调用还可能无法匹配现有允许规则，需要另行核对实际命令和审批范围。不能仅凭审批申请中的“默认沙箱阻止 SSH 传输”等说明文字判定具体失败原因。

2026-10-05 完成不落盘的文本传输验证：直接启动普通 SSH 命令，向其标准输入发送一行 ASCII 测试文本；服务器在进入项目根目录后读取该行并返回字节计数 `25`，SSH 正常退出。执行调用未显式请求额外沙箱权限，但用户随后提供了“Send input to an existing terminal”审批提示，说明给沙箱外终端发送输入还可能触发独立审核。工具执行成功不能证明整个传输过程没有出现审批弹窗；本次只确认数据收发成功，不能认定终端输入已获得长期免审批授权。未进行二进制文件传输或其他封装调用的验证。

上述终端输入提示还说明：终端在本地沙箱外启动，不经过受管网络代理；显示的工作目录是启动目录，平台不保证终端当前目录和状态仍相同。这是平台对终端输入的权限说明，不表示服务器项目目录检查失败，也不取消项目行为约束。发送输入前仍须确认目标会话、当前执行程序和项目范围。

若需要让其他符合条件的权限申请交给自动审核，可在应用的权限菜单选择 **Approve for me**。按 [官方权限模式说明](https://learn.chatgpt.com/docs/permission-modes)，若菜单未显示该选项，可在 **Settings > General > Permissions** 中启用 **Auto-review**，再为当前聊天选择该模式；选项是否可用仍受平台和组织配置影响。自动审核保留原沙箱边界，由审核代理判断是否允许请求，不能保证每次通过。匹配现有 SSH 允许规则的启动命令可直接执行；终端输入等后续工具操作须按各自的审批要求处理。

本项目已在 [`.codex/config.toml`](../.codex/config.toml) 设置：

```toml
approval_policy = "on-request"
approvals_reviewer = "auto_review"
```

按 [官方配置说明](https://learn.chatgpt.com/docs/config-file/config-advanced)，这两个设置把符合条件的交互式审批交给自动审核，保留现有沙箱边界；适用范围是本项目的符合条件审批，并不限于 SSH。配置未改动工作区隔离规则。项目配置仅在项目受信任时加载，仍受更高优先级的运行设置和管理员要求约束，见 [配置加载说明](https://learn.chatgpt.com/docs/config-file/config-basic)。当前会话是否即时采用新设置须实测；若仍出现人工授权弹窗，重新加载项目或重启应用后再验证，并确认当前聊天选择 **Approve for me**。

2026-10-05 写入该配置后首次复测：普通 SSH 连接正常返回 `/cluster/home/zengy/phd-research`；新建 SSH 终端并发送同一行测试文本，服务器返回字节计数 `25`，进程正常退出。用户确认终端输入仍显示“拒绝／允许一次”弹窗，因此该次没有实现终端输入的自动审核。随后平台已明确提供当前会话的 `approvals_reviewer = "auto_review"` 运行设置；再新建 SSH 终端，先输出服务器项目目录，再读取同一行测试文本，仍正常返回 `25` 并退出。用户确认后一项测试没有需要其点击的弹窗，因此本次 SSH 终端输入已验证可在自动审核模式下完成，无需人工确认。两次测试均未写入本地或服务器文件。工具未单独返回审核判定记录，不能据此断定所有后续终端操作都会自动通过；具体输入仍受各自审批要求和审核判断约束。

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
