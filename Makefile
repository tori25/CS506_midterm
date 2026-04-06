PYTHON = python3
NOTEBOOK = notebooks/modeling.ipynb

.PHONY: run clean eda all install help submit

run:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute $(NOTEBOOK) --inplace

clean:
	rm -f submissions/*.csv
	rm -f *.obj
	rm -f assets/*.png

eda:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute notebooks/eda.ipynb --inplace

all: eda run

install:
	pip install -r requirements.txt

help:
	@echo "Available commands:"
	@echo "make run      -> run modeling notebook"
	@echo "make eda      -> run EDA notebook"
	@echo "make clean    -> remove submissions, models, and generated plots"
	@echo "make install  -> install dependencies"

submit: run
	@echo "Submission file created in submissions/"
