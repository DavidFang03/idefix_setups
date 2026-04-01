import ffmpeg
import os
import matplotlib.pyplot as plt
import glob
import matplotlib.image as mpimg
from pathlib import Path
from multiprocessing import Pool


folder1_pattern = "/home/dp316/dp316/dc-fang1/IdefixRuns/AOWind/frames/AOw_Rm10_smaller_1000x1024/global/data*.png"
folder2_pattern = "/home/dp316/dp316/dc-fang1/IdefixRuns/AOWind/frames/AOw_Rm10_1536x1024/global/zoom*.png"

sidebyside_name = "100vs20"
outfolder_path = (
    Path("/home/dp316/dp316/dc-fang1/IdefixRuns/sidebyside") / sidebyside_name
)
try:
    os.makedirs(outfolder_path / "frames")
except:
    pass

list1 = sorted(glob.glob(folder1_pattern))
list2 = sorted(glob.glob(folder2_pattern))


def sidebyside(ii):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(16, 9))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0)
    ax1.axis("off")
    ax2.axis("off")
    img1 = mpimg.imread(list1[ii])
    img2 = mpimg.imread(list2[ii])

    ax1.imshow(img1)
    ax2.imshow(img2)
    fig.savefig(outfolder_path / f"{ii:04}.png", dpi=300)
    plt.close(fig)
    print(f"So far so good {ii:<4}")


skip_to_movie = True
if not skip_to_movie:
    with Pool(16) as pool:
        pool.map(sidebyside, range(min(len(list1), len(list2))))

fps = 10
pattern_png = f"{outfolder_path / '*.png'}"
movie_path = str(outfolder_path / f"{sidebyside_name}.mp4")
print(pattern_png)
ffmpeg.input(pattern_png, pattern_type="glob", framerate=fps).output(
    movie_path,
    vcodec="libx264",
    crf=18,
    preset="medium",
    r=fps,
    pix_fmt="yuv420p",
    movflags="faststart",
).overwrite_output().run()
print(f"[OK] {movie_path}")
