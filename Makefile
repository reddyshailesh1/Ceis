# ============================================================
#  Makefile  —  Clinical Evidence Intelligence System (CEIS)
# ============================================================

PYTHON   ?= python
CONFIG   ?= config/train_config.yaml
API_PORT ?= 8000
UI_PORT  ?= 8501

.PHONY: install data quick train eval hyperopt serve ui notebook clean lint

install:
	pip install -r requirements.txt

# ---- data pipeline -------------------------------------------
data:
	$(PYTHON) main.py data download
	$(PYTHON) main.py data clean
	$(PYTHON) main.py data split

# ---- training ------------------------------------------------
quick:
	$(PYTHON) main.py train --config $(CONFIG) --out runs/quick --limit 200 --epochs 1

train:
	$(PYTHON) main.py train --config $(CONFIG) --out runs/ceis_seed0 --seed 0
	$(PYTHON) main.py train --config $(CONFIG) --out runs/ceis_seed1 --seed 1 --device cuda:1

hyperopt:
	$(PYTHON) main.py hyperopt --config $(CONFIG) --trials 20 --out runs/hyperopt

# ---- evaluation -----------------------------------------------
eval:
	$(PYTHON) main.py eval benchmark
	$(PYTHON) main.py eval calibration
	$(PYTHON) main.py eval ood
	$(PYTHON) main.py eval report

# ---- application ---------------------------------------------
serve:
	$(PYTHON) main.py serve --port $(API_PORT)

ui:
	streamlit run app/frontend/streamlit_app.py --server.port $(UI_PORT)

# ---- notebooks ------------------------------------------------
notebook:
	jupyter notebook notebooks/

# ---- single inference ----------------------------------------
infer:
	$(PYTHON) main.py infer \
	    --claim "Patient has elevated creatinine." \
	    --evidence "Labs were notable for Cr 1.7 (baseline 0.5)."

# ---- code quality --------------------------------------------
lint:
	ruff check . || flake8 .

# ---- cleanup -------------------------------------------------
clean:
	rm -rf runs/ benchmark_out/ reports/__pycache__ data/processed/ data/raw/
	find . -type d -name __pycache__ -exec rm -rf {} +
