#!/bin/bash

echo "Checking Types (mypy)"
mypy mbas --namespace-packages --no-strict-optional
type_err=$?

echo "Checking Formatting (black)"
black mbas --check
format_err=$?

echo "Checking Flake8"
flake8 mbas --show-source
flake_err=$?

if [[ $format_err -eq 0 && $flake_err -eq 0 && $type_err -eq 0 ]]; then
    exit 0
else
    exit 1
fi