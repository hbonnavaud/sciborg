from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from utils.sys_fun import create_dir


def get_red_green_color(value):
    """
    Retourne une couleur correspondant à un gradient entre rouge (0) et vert (1) pour une valeur donnée entre 0 et 1
    :param value: valeur entre 0 et 1 définissant la couleur à récupérer
    """
    value = value * 2 - 1
    if value >= 0:
        red = hex(int((1 - value) * 255))[2:]
        green = hex(255)[2:]
        blue = hex(int((1 - value) * 255))[2:]
    else:
        red = hex(255)[2:]
        green = hex(int((1 + value) * 255))[2:]
        blue = hex(int((1 + value) * 255))[2:]
    red = "0" + red if len(red) == 1 else red
    green = "0" + green if len(green) == 1 else green
    blue = "0" + blue if len(blue) == 1 else blue
    res = "#" + red + green + blue
    return res


def generateVideo(images, directory, filename):
    if len(filename) < 4 or filename[-4:] != ".mp4":
        filename += ".mp4"

    create_dir(directory)
    video = ImageSequenceClip(images, fps=30)
    video.write_videofile(directory + filename)