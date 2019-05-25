.PHONY: features trained-model score-model evaluate-model test clean-tests clean-env clean-pyc

pennylane-env/bin/activate: requirements.txt
	test -d pennylane-env || virtualenv pennylane-env
	. pennylane-env/bin/activate; pip install -r requirements.txt
	touch pennylane-env/bin/activate

venv: pennylane-env/bin/activate

data/features_processed.csv: src/generate_features.py
	python src/generate_features.py --config=config/config.yml --output=data/features_processed.csv

features: data/features_processed.csv

models/risk-prediction.pkl: data/features_processed.csv src/train_model.py
	python src/train_model.py --config=config/config.yml --input=data/features_processed.csv --output=models/risk-prediction.pkl

trained-model: models/risk-prediction.pkl

models/risk-test-scores.csv: src/score_model.py
	python src/score_model.py --config=config/config.yml --output=models/risk-test-scores.csv

score-model: models/risk-test-scores.csv

models/model-evaluation.csv: src/evaluate_model.py
	python src/evaluate_model.py --config=config/config.yml --output=models/model-evaluation.csv

evaluate-model: models/model-evaluation.csv

test:
	pytest src/test.py

clean-tests:
	rm -rf .pytest_cache
	rm -r test/model/test/
	mkdir test/model/test
	touch test/model/test/.gitkeep

clean-env:
	rm -r pennylane-env

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	rm -rf .pytest_cache

all: features trained-model score-model evaluate-model 
