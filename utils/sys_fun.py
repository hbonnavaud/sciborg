import inspect
import os
import shutil
import datetime
import cv2
import numpy as np
from PIL import Image


def empty_dir(directory_path: str, send_to_trash=True):
    if send_to_trash:
        try:
            import send2trash
        except:
            raise ModuleNotFoundError("Install module send2trash in order to use utils.empty_dir function, "
                                      "or add the argument send_to_trash=False.")

    directory_path = os.path.abspath(directory_path)
    if not directory_path.endswith(os.path.sep):
        directory_path += os.path.sep

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if send_to_trash:
            try:
                send2trash.send2trash(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        else:
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def create_dir(directory_name: str, replace=False):

    directory_path = os.path.abspath(directory_name)
    directory_path += os.path.sep

    if os.path.isdir(directory_path):
        # Directory already exists
        if replace:
            shutil.rmtree(directory_path)
        else:
            return

    dir_parts = directory_path.split(os.sep)
    directory_to_create = ""
    for part in dir_parts:
        directory_to_create += part + os.sep
        if not os.path.isdir(directory_to_create):
            try:
                os.mkdir(directory_to_create)
            except FileNotFoundError:
                print("failed to create dir " + str(directory_to_create))
                raise Exception


def generate_video(images, output_directory: str, filename, convert_to_bgr=True):
    """
    generate a video from the given list of images, and save them at location output_directory/filename.mp4.
    @param images: List of images as numpy array of pixels. For each image, the expected shape is width * height * 3.
        images[n][-1] is expected to be a list of rgb pixels. But BGR pixels are accepted if convert_to_bgr is set to
        false.
    @param output_directory: A path. A '/' is added at the end if there's none in the given path.
    @param filename: a filename. Should not contain "/" characters or '.' except for the extension. If no '.' is found
        (aka no extension) a ".mp4" is added at the end.
    @param convert_to_bgr: (boolean) If True (default value), the colors are considered as RGB and are converted to BGR
        (which is the default opencv standard, don't ask me why).
    """
    directory_path = os.path.abspath(output_directory)
    directory_path += os.path.sep

    # Convert image colors
    if convert_to_bgr:
        images = [cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR) for img in images]

    if output_directory[-1] != os.sep:
        output_directory += os.sep

    # Verify filename
    if len(filename) < 4 or filename[-4:] != ".mp4":
        filename += ".mp4"
    assert len(filename.split(".")) == 2

    create_dir(output_directory)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, channels = images[0].shape
    fps = 30

    out = cv2.VideoWriter(output_directory + filename, fourcc, fps, (width, height))
    for image_data in images:
        image_data = image_data.astype(np.uint8)
        out.write(image_data)
    out.release()


def save_image(image: np.ndarray, output_directory: str, file_name: str, extension: str = "png"):
    directory_path = os.path.abspath(output_directory)
    directory_path += os.path.sep

    if output_directory[-1] != os.sep:
        output_directory += os.sep
    if not os.path.isdir(output_directory):
        print("directory ", output_directory, " not found", sep="")
        raise FileNotFoundError("Directory ", output_directory, " not found. Hint: directory without \"/\" at the beginning "
                                "will be considered as relative path. Add \"/\" at the beginning if your path is "
                                "absolute, and remove it if its not.")
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    create_dir(output_directory)
    if not file_name.endswith("." + extension):
        if len(file_name.split(".")) > 1:
            file_name = "".join(file_name.split(".")[:-1])  # Remove the last extension
        assert len(file_name.split(".")) == 1
        file_name += "." + extension
    image.save(output_directory + file_name)
