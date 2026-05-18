# Setup

### Install GNU Make

Linux (Debian/Ubuntu):

```bash
sudo apt update
sudo apt install build-essential make
```

Windows:

- Recommended: Use WSL (Windows Subsystem for Linux) and install Make inside the Linux distro.
- Alternatively, install via Chocolatey or Scoop: `choco install make` or `scoop install make`

## Project setup

Run:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -e .

python scripts/config.py
python scripts/etl.py
```

If you prefer to use `make`, you can run:

```bash
make config
make etl
```
