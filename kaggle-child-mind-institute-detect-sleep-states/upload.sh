#!/bin/bash
# 1. Create symbolic link to files which you want to upload 
# > ln -s ../../infer.py .
# > ln -s ../../src ./src
# 2. Execut the following commands


kaggle datasets version -p . -m "updated" --dir-mode tar
