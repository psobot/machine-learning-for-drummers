.PHONY: all test clean pip

all: pip features.json
	@echo "Training and testing model..."
	python3 classifier.py $<
	@echo "Done!"

pip: requirements.txt
	@echo "Installing requirements..."
	@python3 -m pip install -r $< >/dev/null

features.json: feature_extract.py
	@echo "Extracting features..."
	python3 feature_extract.py > $@

clean:
	rm features.json