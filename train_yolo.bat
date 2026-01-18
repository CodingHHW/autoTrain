@echo off

REM 设置编码为UTF-8
chcp 65001 >nul

REM YOLO训练启动脚本
REM 直接使用虚拟环境Python执行训练命令

echo 正在启动YOLO训练...

REM 设置工作目录为脚本所在目录
cd /d "%~dp0"

REM 虚拟环境Python路径和YOLO训练命令
"C:\ProgramData\miniconda3\envs\yolo\Scripts\yolo.exe" detect train data=D:\autoTrain\config\datasets\me.yaml model=D:\autoTrain\modules\yolo11n.pt epochs=30 patience=20 batch=-1 workers=8 cache=True amp=True hsv_h=0.015 hsv_s=0.4 hsv_v=0.3 degrees=0 translate=0

echo YOLO训练完成！
echo 按任意键退出...
pause > nul
