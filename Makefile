# Windows : utilisez directement .\.venv\Scripts\python.exe si Make n'est pas installé.
# Activez le venv avant make : sinon python désigne le Python global.
.PHONY: data clean series eda features folds test test-run dashboard forecast kernel cv

kernel:
	python -c "import sys; assert sys.prefix != sys.base_prefix, 'activez le venv'"
	python -m ipykernel install --user --name retail-forecast --display-name "Python 3.12 (retail-forecast)"

data:
	python scripts/make_dataset.py

clean:
	python scripts/run_cleaning.py

series:
	python scripts/build_series.py

eda:
	python scripts/build_eda.py
	python scripts/create_eda_notebook.py
	python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=retail-forecast notebooks/02_eda.ipynb

features:
	python scripts/build_features.py

folds:
	python scripts/build_folds.py

cv:
	python scripts/run_cv.py

test:
	python -m pytest

test-run:
	python scripts/run_test.py

dashboard:
	python -m streamlit run app/dashboard.py

forecast:
	python scripts/export_forecast.py
