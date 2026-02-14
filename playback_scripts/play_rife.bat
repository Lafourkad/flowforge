@echo off
REM FlowForge Real-Time RIFE Playback via mpv + VapourSynth
REM Usage: play_rife.bat "path\to\video.mkv"

set VIDEO=%~1
if "%VIDEO%"=="" (
    echo Usage: play_rife.bat "video_file"
    exit /b 1
)

set MPV="C:\Program Files\SVP 4\mpv64\mpv.exe"
set SCRIPT=C:\Users\Kad\Desktop\FlowForge\flowforge_rife.vpy

%MPV% --no-config --hwdec=no --vf=vapoursynth=[file=%SCRIPT%] %VIDEO%
