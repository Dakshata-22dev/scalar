---
title: Email Triage Env
emoji: 📬
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Email Triage Env

A FastAPI-based email triage AI environment designed for deployment on Hugging Face Spaces.

## Endpoints

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /docs`

## Local Run

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
