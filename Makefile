.PHONY: install test smoke proposed baselines report clean

install:
	pip install -r requirements.txt

test:
	python3 -m compileall -q src comparison_models scripts tests
	python3 -m unittest tests.smoke_test

smoke:
	python3 scripts/run_beluga_forecastnet.py --smoke --no-save-model --log-level INFO

proposed:
	python3 scripts/run_beluga_forecastnet.py --log-level INFO

baselines:
	python3 scripts/run_optuna_baselines.py --log-level INFO

report:
	python3 scripts/generate_results_report.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
