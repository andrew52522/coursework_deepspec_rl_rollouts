# 00 — Подготовка чистого WSL2-ноутбука с RTX 4070 к GRPO preflight

**Актуально:** 29 сентября 2026  
**Следующий документ:** `plan_grpo_1x4070_then_2x4070.md` / этап полного GRPO smoke.  
**Назначение этого документа:** довести полностью чистый ноутбук до состояния, в котором можно безопасно начинать устанавливать и запускать pinned `verl`, не смешав драйверы, CUDA, Python-окружения, кэши, Git и WSL.

---

## 1. Контекст проекта

Курсовая исследует live GRPO для Qwen с rollout через inference engine и далее speculative decoding. Финальная целевая среда — 2× RTX 5090 или 2× H100, Qwen3-4B, `verl`, FSDP2, vLLM, GSM8K и замороженные draft heads.

Ноутбук с RTX 4070 используется **не для финальных метрик**, а как инженерный preflight:

1. проверить Linux/GPU/Python stack;
2. проверить GSM8K preprocessing;
3. проверить полный GRPO цикл на маленькой модели;
4. проверить backward, optimizer update и weight synchronization;
5. проверить checkpoint/resume;
6. только после этого переносить тот же код на дорогой сервер.

Для 1×4070 основной preflight target:

- GPU: RTX 4070 Laptop, 8 GB VRAM;
- RAM хоста: 64 GB;
- ОС хоста: Windows 10 Pro;
- Linux: Ubuntu 24.04 в WSL2;
- модель: `Qwen/Qwen3-0.6B`;
- speculative decoding: выключен, `SPEC_METHOD=none`;
- training backend: FSDP2;
- rollout backend: vLLM;
- dataset: GSM8K;
- pinned verl commit: `6093e007cc341973c9d9a6fb3867a85976c7c458`.

---

# 2. Физическая схема машины

Рабочая схема сейчас:

```text
Mac
  |
  | local network + SSH
  v
Windows 10 Pro laptop
  |
  | Windows OpenSSH / existing forwarding
  v
Ubuntu 24.04 in WSL2
  |
  v
Linux shell + project + Python/uv/verl
  |
  v
RTX 4070 through CUDA-on-WSL
```

Основная разработка выполняется **внутри Ubuntu/WSL2** через SSH с Mac.

Windows нужен как:

- физический host;
- владелец NVIDIA display driver;
- WSL2 VM host;
- место для глобальных WSL-настроек `.wslconfig`;
- сетевой gateway для уже настроенного SSH.

---

# 3. Главное правило CUDA в WSL

## Не устанавливать Linux NVIDIA display driver внутрь Ubuntu

В WSL2 GPU-драйвер предоставляет **Windows NVIDIA driver**. В Linux виден proxy/stub CUDA driver.

Поэтому внутри Ubuntu нельзя выполнять такие действия без отдельной обоснованной причины:

```text
sudo apt install nvidia-driver-...
sudo ubuntu-drivers install
sudo apt install cuda-drivers
sudo apt install cuda
```

Также нельзя следовать обычной Ubuntu-инструкции NVIDIA driver installation как для bare-metal Linux.

NVIDIA прямо указывает: Windows driver — единственный display driver, необходимый для CUDA on WSL; Linux display driver внутри WSL устанавливать не нужно.

Если позже понадобится CUDA Toolkit / `nvcc` для сборки конкретного extension, выбирать WSL-compatible **toolkit-only** вариант и не устанавливать Linux driver.

Для первого этапа GRPO мы **не ставим CUDA Toolkit вручную вообще**, пока pinned `verl` environment не покажет, что он действительно нужен.

Официальный источник:
https://docs.nvidia.com/cuda/wsl-user-guide/

---

# 4. Сначала зафиксировать Windows/WSL baseline

Эти команды выполняются в **Windows PowerShell**, а не в Ubuntu.

Попросить ChatGPT объяснять каждую команду перед выполнением и не выполнять массовые изменения одновременно.

## 4.1. Проверить Windows

```powershell
winver
Get-ComputerInfo | Select-Object WindowsProductName,WindowsVersion,OsBuildNumber
```

Нужно сохранить:

- Windows 10 Pro;
- build number.

WSL требует Windows 10 версии 2004 / build 19041 или новее. Для современного GPU tooling желателен максимально обновлённый поддерживаемый Windows 10 build.

Источник:
https://learn.microsoft.com/windows/wsl/install

## 4.2. Проверить WSL

```powershell
wsl --status
wsl --version
wsl -l -v
```

Нужно убедиться:

- Ubuntu 24.04 существует;
- VERSION = 2;
- выбран правильный distro;
- WSL достаточно новый, чтобы поддерживать современный WSL2/systemd workflow.

Если `wsl --version` не поддерживается, сначала разбираться с версией WSL, а не ставить ML stack.

## 4.3. Обновить WSL только после фиксации baseline

Если baseline сохранён и WSL устарел:

```powershell
wsl --update
```

После обновления повторить:

```powershell
wsl --status
wsl --version
wsl -l -v
```

Источник:
https://learn.microsoft.com/windows/wsl/basic-commands

---

# 5. Проверить Windows NVIDIA driver

На Windows:

```powershell
nvidia-smi
```

Сохранить:

- GPU name;
- driver version;
- reported CUDA compatibility;
- VRAM.

Важно: поле `CUDA Version` в `nvidia-smi` означает поддерживаемую драйвером CUDA API/runtime compatibility, а не версию Python PyTorch runtime, которую мы потом будем использовать.

Если Windows `nvidia-smi` не видит RTX 4070 — **не продолжать в Ubuntu**. Сначала чинить Windows driver.

---

# 6. Проверить GPU из Ubuntu/WSL

Теперь подключиться с Mac по существующему SSH и работать внутри Ubuntu.

Выполнить:

```bash
whoami
hostname
pwd
cat /etc/os-release
uname -a
cat /proc/version
nvidia-smi
nvidia-smi -L
```

Ожидается:

- Ubuntu 24.04;
- Microsoft/WSL kernel;
- RTX 4070 видна;
- `nvidia-smi` работает внутри WSL.

Если команда `nvidia-smi` не найдена, сначала проверить:

```bash
ls -l /usr/lib/wsl/lib/nvidia-smi
echo "$PATH"
```

Не решать отсутствие команды установкой Linux NVIDIA driver.

---

# 7. Проверить systemd

Ubuntu 24.04 в современном WSL обычно использует systemd, но это нужно проверить:

```bash
ps -p 1 -o comm=
systemctl is-system-running || true
systemctl status ssh --no-pager || true
```

Если PID 1 = `systemd`, всё нормально.

Если systemd не включён, сначала выяснить реальное состояние `/etc/wsl.conf` и версию WSL. Не переписывать конфиг наугад.

Официальная документация:
https://learn.microsoft.com/windows/wsl/systemd/

---

# 8. Очень важный WSL memory вопрос

Windows host имеет 64 GB RAM.

По умолчанию WSL2 VM получает лимит примерно **50% физической RAM Windows**. Для 64 GB это около 32 GB. Для обычной разработки этого достаточно, но для будущего FSDP/optimizer/reference CPU offload 32 GB может стать искусственным ограничением.

Сначала внутри Ubuntu записать:

```bash
free -h
grep MemTotal /proc/meminfo
nproc
df -h /
```

Затем на Windows проверить, существует ли:

```text
%UserProfile%\.wslconfig
```

## Рекомендуемый стартовый вариант для dedicated ML laptop

Это **инженерная рекомендация**, не обязательное значение Microsoft:

```ini
[wsl2]
memory=52GB
swap=16GB
localhostForwarding=true
```

Почему не 64 GB:

- Windows тоже нужна RAM;
- NVIDIA driver и Windows services живут на host;
- оставляем запас, чтобы Windows не начал тяжело paging.

Если Windows нестабилен — уменьшить WSL memory, например до 48 GB.

Не ограничивать `processors`, пока нет причины: по умолчанию WSL может использовать все logical processors.

После изменения `.wslconfig` требуется полный shutdown WSL:

```powershell
wsl --shutdown
```

**Внимание:** это оборвёт SSH-сессию Mac → WSL. После shutdown нужно снова запустить Ubuntu на Windows host, затем подключиться по SSH.

После перезапуска снова выполнить:

```bash
free -h
nproc
```

Официальная документация:
https://learn.microsoft.com/windows/wsl/wsl-config

---

# 9. Где должны лежать проект, модели и кэши

## Не работать из /mnt/c

ML repo, `.venv`, uv cache, Hugging Face cache и training files должны находиться в Linux filesystem WSL, например:

```text
/home/<user>/coursework/
/home/<user>/.cache/
/home/<user>/datasets/
```

а не:

```text
/mnt/c/Users/...
```

Microsoft рекомендует хранить Linux-проекты в WSL filesystem, если инструменты выполняются в Linux. Работа с build/package workloads через `/mnt/c` имеет дополнительный filesystem overhead.

Источник:
https://learn.microsoft.com/windows/wsl/filesystems

---

# 10. Проверить реальный свободный диск

В Ubuntu:

```bash
df -h /
df -h ~
du -sh ~ 2>/dev/null || true
```

На Windows также проверить свободное место физического SSD, где находится `ext4.vhdx`.

Для проекта желательно иметь **существенный запас**, потому что место займут:

- uv package cache;
- PyTorch/CUDA wheels;
- vLLM;
- verl virtual environment;
- Hugging Face weights;
- datasets;
- training logs;
- temporary files;
- checkpoints.

Не хранить финальные 4B checkpoints и большие model caches в Git.

Официально WSL2 хранит distro в ext4 VHDX:
https://learn.microsoft.com/windows/wsl/disk-space

---

# 11. Обновить Ubuntu system packages

Только после того, как GPU уже видна и baseline сохранён.

```bash
sudo apt update
sudo apt upgrade -y
```

Если обновление просит перезапуск важных WSL/system services — закончить upgrade, затем при необходимости перезапустить WSL контролируемо.

После этого:

```bash
sudo apt autoremove -y
```

Не удалять руками CUDA/WSL libraries из `/usr/lib/wsl`.

---

# 12. Установить только базовые системные инструменты

```bash
sudo apt install -y \
  build-essential \
  git \
  curl \
  wget \
  ca-certificates \
  gnupg \
  lsb-release \
  pkg-config \
  unzip \
  zip \
  jq \
  ripgrep \
  rsync \
  tmux \
  htop \
  tree \
  openssh-client \
  iproute2 \
  iputils-ping
```

Это базовые Linux tools. Они не задают PyTorch/CUDA/vLLM версии.

Проверить:

```bash
git --version
curl --version
rg --version
tmux -V
python3 --version
```

Ubuntu 24.04 обычно имеет Python 3.12. Системный Python нужен как базовый инструмент, но GPU ML packages мы не ставим в него через `sudo pip`.

---

# 13. Что пока НЕ устанавливать

До клонирования pinned `verl` не устанавливать:

```text
conda / miniconda только ради проекта
pip install torch
pip install vllm
pip install flash-attn
pip install ray
pip install transformers -U
pip install verl
sudo apt install cuda
sudo apt install nvidia-driver-*
```

Причина: coursework должен воспроизводиться через одну зафиксированную dependency graph, а не через накопленный глобальный Python environment.

---

# 14. Настроить Git identity внутри WSL

Проверить:

```bash
git config --global user.name
git config --global user.email
```

Если пусто:

```bash
git config --global user.name "YOUR NAME"
git config --global user.email "YOUR GITHUB EMAIL"
```

Проверить:

```bash
git config --global --list
```

Email желательно использовать тот, который привязан к GitHub, либо GitHub noreply email.

Источник:
https://docs.github.com/en/get-started/git-basics/set-up-git

---

# 15. Настроить GitHub authentication внутри WSL

Важно различать два SSH:

1. Mac → Windows/WSL: удалённый доступ к ноутбуку;
2. WSL → GitHub: аутентификация Git для clone/push.

Это **разные** задачи и ключи могут быть разными.

## Рекомендуемый простой путь: GitHub CLI + HTTPS

Установить `gh` по официальной инструкции GitHub CLI для Ubuntu, затем:

```bash
gh auth login
```

Выбрать:

- GitHub.com;
- HTTPS;
- authenticate Git with GitHub credentials = Yes;
- browser/device login.

После этого:

```bash
gh auth status
git ls-remote https://github.com/andrew52522/coursework_deepspec_rl_rollouts.git
```

GitHub рекомендует GitHub CLI/Git Credential Manager для credential caching при HTTPS.

Источники:
https://docs.github.com/en/authentication
https://docs.github.com/en/get-started/git-basics/caching-your-github-credentials-in-git

## Альтернатива: отдельный SSH key внутри WSL

Можно создать отдельный ключ в `~/.ssh` Ubuntu и добавить public key в GitHub. Не копировать приватный ключ Mac без необходимости.

---

# 16. Клонировать coursework repo ВНУТРЬ Linux filesystem

Создать нормальную структуру:

```bash
mkdir -p ~/coursework
cd ~/coursework

git clone https://github.com/andrew52522/coursework_deepspec_rl_rollouts.git
cd coursework_deepspec_rl_rollouts

git remote -v
git status
git branch --show-current
```

Ожидается remote:

```text
origin  https://github.com/andrew52522/coursework_deepspec_rl_rollouts.git
```

Путь должен выглядеть примерно так:

```text
/home/<user>/coursework/coursework_deepspec_rl_rollouts
```

а не `/mnt/c/...`.

---

# 17. Правила репозитория для курсовой

В Git должны попадать:

- README;
- architecture notes;
- reproducible scripts;
- launch scripts;
- config/profile files;
- tiny test fixtures;
- code for metrics;
- verification scripts;
- exact commit hashes/revisions;
- curated small evidence/log excerpts;
- experiment manifests;
- plots/tables for final report.

В Git **не должны** попадать:

- model weights;
- Hugging Face caches;
- uv/pip caches;
- `.venv`;
- GSM8K parquet;
- raw huge rollout dumps;
- checkpoints;
- optimizer states;
- Ray temp files;
- WandB offline cache;
- SSH private keys;
- API tokens;
- `.env`;
- large binary artifacts.

Большие experimental artifacts должны лежать на локальном диске/сервере и описываться в manifest: путь, checksum, модель/revision, run ID.

---

# 18. Базовая структура coursework repo

Целевая структура:

```text
coursework_deepspec_rl_rollouts/
├── README.md
├── CHATGPT_CONTEXT.md
├── .gitignore
├── docs/
│   ├── 00_wsl4070_environment_preflight.md
│   ├── 01_grpo_1x4070_then_2x4070.md
│   ├── architecture/
│   └── experiments/
├── scripts/
│   ├── prepare_gsm8k.sh
│   └── run_live_grpo.sh
├── tools/
│   └── verify_live_loop.py
├── config/
│   └── profiles/
├── tests/
└── reports/
```

Создавать директории только когда появляется первый реальный файл. Не коммитить пустые директории просто ради структуры.

---

# 19. Установить uv

`verl` использует `uv.lock`, и coursework должен использовать тот же locked environment.

Официальная установка:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Перезапустить shell или загрузить environment, который напечатает installer.

Проверить:

```bash
uv --version
which uv
```

Не делать глобальный `uv pip install torch`.

Официальная документация:
https://docs.astral.sh/uv/

---

# 20. Настроить кэши предсказуемо

Для первого ноутбука удобно заранее определить директории:

```bash
mkdir -p ~/cache/huggingface
mkdir -p ~/cache/uv
mkdir -p ~/datasets
mkdir -p ~/runs
```

В shell profile можно позже добавить:

```bash
export HF_HOME="$HOME/cache/huggingface"
export UV_CACHE_DIR="$HOME/cache/uv"
```

Но сначала проверить, что paths не находятся на `/mnt/c`.

Не добавлять токены Hugging Face/GitHub в Git-tracked files.

Если модель публичная, Hugging Face token вообще не нужен для первого Qwen3-0.6B smoke.

---

# 21. Зафиксировать полный preflight passport

Создать локальный файл, который можно потом приложить к issue/experiment notes:

```bash
mkdir -p ~/coursework/preflight

{
  echo "=== DATE ==="
  date -Is

  echo "=== USER/HOST ==="
  whoami
  hostname
  pwd

  echo "=== OS ==="
  cat /etc/os-release
  uname -a
  cat /proc/version

  echo "=== WSL ==="
  printf 'WSL_DISTRO_NAME=%s\n' "$WSL_DISTRO_NAME"
  printf 'WSL_INTEROP=%s\n' "$WSL_INTEROP"

  echo "=== GPU ==="
  nvidia-smi
  nvidia-smi -L

  echo "=== RAM/CPU ==="
  free -h
  nproc
  lscpu | sed -n '1,30p'

  echo "=== DISK ==="
  df -h /
  df -h ~

  echo "=== TOOLS ==="
  git --version
  curl --version | head -1
  python3 --version
  uv --version || true

  echo "=== NETWORK ==="
  ip -br addr
  ip route
} | tee ~/coursework/preflight/passport_wsl4070.txt
```

Этот файл содержит machine metadata, поэтому перед публичным commit его нужно открыть и проверить, нет ли нежелательных hostname/IP/username деталей.

Лучше хранить полный passport локально, а в Git — sanitized summary.

---

# 22. Проверить SSH workflow с Mac

Так как Mac уже подключается к WSL, нужно только проверить устойчивость:

```bash
echo "$SSH_CONNECTION"
echo "$SSH_TTY"
who
```

Для долгих задач использовать `tmux`:

```bash
tmux new -s coursework
```

Отсоединение:

```text
Ctrl-b d
```

Возврат:

```bash
tmux attach -t coursework
```

Это важно: разрыв Mac Wi‑Fi/SSH не должен убивать скачивание environment или training run.

---

# 23. Особенность WSL networking на Windows 10

На Windows 10 WSL2 обычно работает в NAT-based networking mode.

Не строить проект вокруг внутреннего WSL IP `172.x.x.x`: он может меняться при рестарте WSL.

Текущий рабочий доступ Mac → Windows → WSL через уже настроенный SSH/localhost forwarding оставить как есть, пока он стабилен.

Mirrored networking — функция Windows 11 22H2+, поэтому инструкция для Windows 11 не должна механически переноситься на этот Windows 10 laptop.

Источники:
https://learn.microsoft.com/windows/wsl/networking
https://learn.microsoft.com/windows/dev-environment/wsl-interop

---

# 24. Проверить DNS и доступ в интернет из WSL

```bash
getent hosts github.com
getent hosts huggingface.co
curl -I https://github.com
curl -I https://huggingface.co
```

Нужно убедиться, что WSL сможет:

- clone GitHub repos;
- скачать uv packages;
- скачать Hugging Face models/datasets.

Если университетская сеть использует proxy/VPN, сначала фиксировать проблему сети, а не менять pip/uv registries случайным образом.

---

# 25. Проверить время и сертификаты

```bash
date
timedatectl || true
dpkg -l ca-certificates | cat
```

Сильно неправильное время ломает TLS и package downloads.

---

# 26. Первый Git commit с environment documentation

После клонирования repo:

1. проверить `.gitignore`;
2. добавить/обновить README;
3. добавить этот preflight document;
4. добавить `CHATGPT_CONTEXT.md`;
5. сделать понятный commit;
6. push в `main`.

Пример commit message:

```text
docs: add WSL 4070 environment preflight
```

После push открыть GitHub web UI и проверить, что:

- README читается;
- нет секретов;
- нет model files;
- нет cache;
- документация понятна без локального контекста.

---

# 27. Только теперь клонировать pinned verl

`verl` должен жить **рядом**, а не как скопированный код внутри coursework repo.

Рекомендуемая layout:

```text
~/coursework/
├── coursework_deepspec_rl_rollouts/
└── third_party/
    └── verl/
```

Создать:

```bash
mkdir -p ~/coursework/third_party
cd ~/coursework/third_party

git clone https://github.com/verl-project/verl.git
cd verl
git checkout 6093e007cc341973c9d9a6fb3867a85976c7c458
git rev-parse HEAD
git status
```

Ожидаемый SHA:

```text
6093e007cc341973c9d9a6fb3867a85976c7c458
```

В coursework repo записываем SHA, но не vendor'им весь verl repo.

---

# 28. Не применять latest verl docs буквально к pinned commit

На дату этого документа latest verl уже имеет другой dependency stack и новые версии vLLM/CUDA.

Для **концептов** latest docs полезны.

Для **точных config keys, launch scripts и dependency versions** source of truth:

1. pinned commit;
2. его `pyproject.toml`;
3. его `uv.lock`;
4. его example script;
5. его конкретный Python code.

Не обновлять pinned commit только потому, что latest docs показывают другую команду.

---

# 29. Первый materialization pinned environment

Из корня pinned verl:

```bash
cd ~/coursework/third_party/verl

uv run --frozen --all-packages --extra vllm --extra fsdp \
  python3 -c "import torch, ray, vllm; print(torch.__version__); print(ray.__version__); print(vllm.__version__)"
```

Первый запуск может долго скачивать packages.

Запускать внутри `tmux`.

Если команда падает:

- сохранить полный error;
- не выполнять случайный `pip install`;
- не обновлять torch/vLLM;
- сначала определить, какой package/driver/kernel/wheel является причиной.

---

# 30. Проверить реальную CUDA из pinned Python environment

После успешных imports:

```bash
cd ~/coursework/third_party/verl

uv run --frozen --all-packages --extra vllm --extra fsdp \
python3 - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("bf16:", torch.cuda.is_bf16_supported())

    x = torch.randn((2048, 2048), device="cuda", dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    print("bf16 matmul ok:", y.shape, y.dtype)
PY
```

Это намного важнее, чем просто `nvidia-smi`.

Успех означает:

- pinned PyTorch импортируется;
- CUDA driver interface работает через WSL;
- RTX 4070 распознаётся;
- реальная BF16 CUDA operation выполняется.

---

# 31. Критерий: preflight завершён

До перехода к полноценному `plan_grpo_1x4070` должны быть выполнены все пункты:

- [ ] Windows 10 build записан;
- [ ] WSL2 version/status записаны;
- [ ] Ubuntu 24.04 работает как WSL2;
- [ ] Windows `nvidia-smi` видит RTX 4070;
- [ ] WSL `nvidia-smi` видит RTX 4070;
- [ ] Linux NVIDIA driver внутри WSL не установлен;
- [ ] systemd проверен;
- [ ] WSL RAM limit проверен и при необходимости поднят;
- [ ] проект расположен в `/home/...`, не `/mnt/c`;
- [ ] есть достаточный свободный SSD;
- [ ] apt base tools установлены;
- [ ] Git identity настроен;
- [ ] WSL → GitHub auth работает;
- [ ] coursework repo клонирован;
- [ ] push в GitHub работает;
- [ ] `uv` установлен;
- [ ] Hugging Face/uv cache directories определены;
- [ ] DNS/HTTPS GitHub/Hugging Face работают;
- [ ] pinned `verl` cloned separately;
- [ ] HEAD равен `6093e007cc341973c9d9a6fb3867a85976c7c458`;
- [ ] `uv run --frozen --all-packages --extra vllm --extra fsdp` материализует environment;
- [ ] torch/ray/vLLM imports работают;
- [ ] PyTorch видит одну GPU;
- [ ] реальная BF16 CUDA operation проходит.

После этого начинается следующий документ: Qwen3-0.6B → GSM8K → 1-step GRPO → 2-step live GRPO → checkpoint/resume.

---

# 32. Как работать с ChatGPT во время настройки

Для каждого шага новый/текущий чат должен работать так:

1. ChatGPT даёт **один логический шаг**, а не 30 непроверенных команд сразу.
2. Перед командой объясняет:
   - что она проверяет/меняет;
   - где её запускать: Windows PowerShell или WSL Bash;
   - какой ожидается результат;
   - опасна ли команда.
3. Пользователь выполняет команду сам.
4. Пользователь присылает полный output.
5. ChatGPT анализирует output и только после этого предлагает следующий шаг.
6. При ошибке не менять сразу несколько компонентов.
7. Любая правка dependency stack должна быть записана в Git/notes.
8. Любой рабочий state фиксируется commit'ом прежде, чем делать рискованное изменение.

---

# 33. Запрещённые сокращения пути для ChatGPT

Если будущий ChatGPT видит ошибку, он **не должен автоматически советовать**:

- `sudo apt install nvidia-driver`;
- `sudo ubuntu-drivers autoinstall`;
- `pip install -U torch`;
- `pip install -U vllm`;
- `pip install flash-attn`;
- заменить pinned verl на latest;
- поменять сразу CUDA + torch + vLLM;
- запускать проект из `/mnt/c`;
- отключать firewall/SSH security целиком;
- складывать tokens/private keys в repo;
- коммитить модели/checkpoints.

Сначала должна быть доказана точная причина ошибки.

---

# 34. Официальные источники

NVIDIA CUDA on WSL:
https://docs.nvidia.com/cuda/wsl-user-guide/

Microsoft WSL installation:
https://learn.microsoft.com/windows/wsl/install

Microsoft WSL commands:
https://learn.microsoft.com/windows/wsl/basic-commands

Microsoft WSL systemd:
https://learn.microsoft.com/windows/wsl/systemd/

Microsoft WSL config / memory:
https://learn.microsoft.com/windows/wsl/wsl-config

Microsoft WSL networking:
https://learn.microsoft.com/windows/wsl/networking

Microsoft WSL filesystem performance:
https://learn.microsoft.com/windows/wsl/filesystems

Microsoft WSL disk space:
https://learn.microsoft.com/windows/wsl/disk-space

GitHub Git setup:
https://docs.github.com/en/get-started/git-basics/set-up-git

GitHub authentication:
https://docs.github.com/en/authentication

uv:
https://docs.astral.sh/uv/

verl installation concepts:
https://verl.readthedocs.io/en/latest/start/install.html

---

# 35. Handoff после preflight

Когда этот документ полностью выполнен, новый ChatGPT-сеанс должен получить:

1. `CHATGPT_CONTEXT.md`;
2. sanitized `passport_wsl4070.txt`;
3. output pinned environment import test;
4. output BF16 CUDA matmul test;
5. git commit SHA coursework repo;
6. pinned verl SHA.

После этого задача чата:

> Перейти к `plan_grpo_1x4070`: Qwen3-0.6B, GSM8K, SPEC_METHOD=none, один GRPO step, затем два live steps с доказанным backward/optimizer/weight sync. Не менять зафиксированный environment без воспроизводимой причины.
