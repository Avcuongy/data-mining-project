# Project setup

## Enviroment

- **Python Version:** Python >= 3.9

## Project setup

Run the following commands in your terminal:

```bash
git clone https://github.com/Avcuongy/data-mining-project.git

cd data-mining-project

python -m venv .venv

.venv\Scripts\Activate.ps1

pip install -r requirements.txt

pip install -e .

python scripts/config.py   # make config
python scripts/etl.py      # make etl
```