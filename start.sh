#!/bin/bash

pip install -r requirements.txt
uvicorn clip_api:app --host 0.0.0.0 --port $PORT