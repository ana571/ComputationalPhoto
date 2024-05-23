# ComputationalPhoto

Computational photography 2024 project 2
ffmpeg -framerate 30 -i frame\_%d.jpg -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" stiched_outdoor.mp4
