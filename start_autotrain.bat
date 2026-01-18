@echo off

REM 设置编码为UTF-8
chcp 65001 >nul

REM autoTrain软件启动脚本

echo 正在启动autoTrain软件...

REM 设置工作目录为脚本所在目录
cd /d "%~dp0"

REM 检查主程序文件是否存在
if not exist "src/main.py" (
    echo 错误：未找到主程序文件 src/main.py
    echo 请确保脚本在autoTrain软件根目录执行
    echo 按任意键退出...
    pause > nul
    exit /b 1
)

REM 使用虚拟环境Python执行主程序
"C:\ProgramData\miniconda3\envs\yolo\python.exe" src/main.py

REM 执行完成后自动退出
