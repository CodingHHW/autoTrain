@echo off

REM YOLO训练启动脚本
REM 打开Anaconda PowerShell终端，切换到yolo虚拟环境，执行训练命令

echo 正在启动YOLO训练...
echo 请稍候，正在打开Anaconda PowerShell终端...

REM 检查Anaconda是否安装
if not exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    echo 错误：未找到Anaconda安装，请检查安装路径
    echo 按任意键退出...
    pause > nul
    exit /b 1
)

REM 使用PowerShell执行命令
PowerShell.exe -Command "& {
    # 激活conda环境
    & '%USERPROFILE%\anaconda3\shell\condabin\conda-hook.ps1'
    # 激活yolo虚拟环境
    conda activate yolo
    # 执行YOLO训练命令
    yolo detect train data=D:\autoTrain\config\datasets\me.yaml model=D:\autoTrain\modules\yolo11n.pt epochs=30 patience=20 batch=-1 workers=8 cache=True amp=True hsv_h=0.015 hsv_s=0.4 hsv_v=0.3 degrees=0 translate=0
}"

echo YOLO训练已启动，训练命令将在Anaconda PowerShell终端中执行
echo 您可以在弹出的终端中查看训练进度
echo 按任意键退出本窗口...
pause > nul
