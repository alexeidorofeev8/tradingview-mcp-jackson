@echo off
echo Запуск TradingView Desktop с CDP на порту 9222...
powershell -Command "Start-Process 'C:\Program Files\WindowsApps\TradingView.Desktop_3.0.0.7652_x64__n534cwy3pjxzj\TradingView.exe' -ArgumentList '--remote-debugging-port=9222'"
if %ERRORLEVEL% NEQ 0 (
    echo Ошибка запуска через WindowsApps. Попробуй скачать TradingView Desktop напрямую:
    echo https://www.tradingview.com/desktop/
    pause
) else (
    echo TradingView запущен. Подожди 5-10 секунд и проверь: tv_health_check
)
