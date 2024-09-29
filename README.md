# Saturn

## Документация

Полная документация доступна по адресу: [https://saturn.talkiiing.ru](https://saturn.talkiiing.ru)

## Разработка

Запуск всего проекта в production-like окружении осуществляется с помощью Compose командой:

`docker compose up --build`

Для отладки рекомендую запускать отдельные компоненты вручную, а внешние сервисы поднять с помощью команды:

`docker compose up --build chroma redis`
