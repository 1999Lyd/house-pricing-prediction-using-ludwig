install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py --line-length 79

lint:
	# pylint --disable=R,C,broad-except,bare-except,consider-using-f-string *.py

test:
	# python -m pytest -vv test_run_auto_train.py

train-ludwig:
	time ludwig train --config house_pricing.yaml --dataset data/train.csv

all: install format lint test