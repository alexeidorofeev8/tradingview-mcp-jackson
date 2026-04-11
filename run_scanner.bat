@echo off
cd /d "d:\projects\trading"
python scanner_alert.py >> logs\scanner_log.txt 2>&1
