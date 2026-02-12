.PHONY: dev test

dev:
	./scripts/run_automom.sh

test:
	pytest backend/tests -q
