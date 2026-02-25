T20 Toss Predictor
==================

[![Stars](https://img.shields.io/github/stars/ama8ine1712/t20-toss-predictor?style=flat&color=70a5fd)](https://github.com/ama8ine1712/t20-toss-predictor/stargazers)
[![Issues](https://img.shields.io/github/issues/ama8ine1712/t20-toss-predictor?style=flat&color=ff7b72)](https://github.com/ama8ine1712/t20-toss-predictor/issues)
![Theme](https://img.shields.io/badge/theme-tokyonight-1a1b27?style=flat)

Machine-learning project to predict T20 cricket toss outcomes using historical match data and engineered features. Large datasets are excluded from version control to keep the repository lightweight and within GitHub size limits.

Features
--------
- Parses historical T20I JSON datasets (Cricsheet format)
- Feature engineering for venue, teams, season, and conditions
- Training and evaluation pipeline
- Simple CLI runner for predictions

Project Structure
-----------------
- run_model.py — run inference/predictions on prepared data
- toss_engine.py — core feature engineering and model logic
- train_winner_only.py — example training pipeline
- requirements.txt — Python dependencies

Setup
-----
1) Create a virtual environment and install dependencies:
   
   python -m venv .venv && .venv\\Scripts\\activate
   pip install -r requirements.txt

2) Obtain datasets (Cricsheet T20I JSON) and place them under data/.
   - Cricsheet: https://cricsheet.org/matches/

Usage
-----
- Train (example):

  python train_winner_only.py

- Predict:

  python run_model.py

Notes
-----
- Large files are ignored via .gitignore (see data/ and winner_data.json).
- For reproducibility, pin dependency versions and document dataset versions.

License
-------
MIT

