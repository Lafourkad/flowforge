# FlowForge Real-Time RIFE — 60fps + Scene Detection
import vapoursynth as vs

core = vs.core

# Load the RIFE plugin
core.std.LoadPlugin(r"C:\Users\Kad\Desktop\FlowForge\vs-plugins\librife.dll")

# Get the video from mpv
clip = video_in

# Scene detection BEFORE color conversion (needs YUV/Gray)
clip = core.misc.SCDetect(clip, threshold=0.100)

# Convert YUV -> RGBS (32-bit float RGB) as required by RIFE
clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

# RIFE interpolation to 60fps
# fps_num/fps_den supersedes factor_num/factor_den
clip = core.rife.RIFE(
    clip,
    model=23,
    fps_num=60,
    fps_den=1,
    model_path=r"C:\Users\Kad\Desktop\FlowForge\vs-plugins\rife-v4.6",
    gpu_id=0,
    gpu_thread=3,
    sc=True,
    tta=False,
    uhd=False,
)

# Convert back to YUV for mpv output
clip = core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

clip.set_output()
