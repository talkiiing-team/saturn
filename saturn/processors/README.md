# Saturn Processors

## Разработка

Локальный запуск воркера конкретной таски осуществляется из корня репозитория командой:

`taskiq worker --log-level DEBUG --reload --workers 1 saturn.libs.broker:broker saturn.processors.tasks.<модуль таски>`
