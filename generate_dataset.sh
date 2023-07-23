#!/bin/bash

echo "Installing requirements"
pip install -qq -r requirements.txt

echo "Generating English dataset"
python ./src/generate_dataset_en.py
