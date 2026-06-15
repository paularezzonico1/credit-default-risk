.PHONY: install run notebook test

install:
	pip install -r requirements.txt

run:
	streamlit run app.py

notebook:
	jupyter nbconvert --to notebook --execute --inplace "notebooks/Credit Default Risk Analysis.ipynb"

test:
	pytest -q
