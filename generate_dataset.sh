#!/bin/bash

echo "Installing requirements"
pip install -qq -r requirements.txt

echo "Generating English dataset"
python3 ./src/generate_dataset_en.py

echo "Generating German dataset"
python3 ./src/generate_dataset_de.py
