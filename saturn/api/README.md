# Saturn API

## Разработка

Локальный запуск Saturn API осуществляется с помощью `uvicorn` из корня репозитория командой:

`uvicorn saturn.api.app:app --reload --log-level=debug --host=0.0.0.0 --port=3000`
