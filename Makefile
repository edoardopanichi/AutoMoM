.PHONY: dev test

dev:
	python run_automom.py

test:
	pytest backend/tests -q
