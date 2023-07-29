import random
from typing import Tuple, Dict, List
import albumentations as A
import numpy as np
import pillow_heif
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont
from albumentations import Compose
from matplotlib import font_manager
from moms_apriltag import TagGenerator3
import argparse

from aug_transform import transform


def page(shape: Tuple[int, int]) -> Image:
    return Image.new('RGB', (shape[0], shape[1]), (255, 255, 255))  # White


def get_font(font_size: int) -> FreeTypeFont:
    font = font_manager.FontProperties(family='arial', weight='regular')
    file = font_manager.findfont(font)
    return ImageFont.truetype(file, font_size)


def gen_tags(transform: Compose, n_tags: int, ids: List[int]) -> Dict[str, Image.Image]:
    tgs = {}
    for i in range(n_tags):
        code = tg.generate(ids[i])
        pill_im = Image.fromarray(code)

        pill_im = pill_im.resize((size_to_pixels, size_to_pixels), Image.NEAREST)

        if pill_im.mode != 'RGB':
            pill_im = pill_im.convert('RGB')

        augmented_image = transform(image=np.array(pill_im))['image']

        tgs[ids[i]] = Image.fromarray(augmented_image.astype('uint8'), 'RGB')
    return tgs


def paint(tags: Dict[str, Image.Image], per_row: int, page_im: Image, draw: ImageDraw):
    __it = iter(tags)

    # text font
    font = get_font(font_size=25)

    for i in range(1, per_row):
        for j in range(1, per_row):
            tag_id = next(__it)
            im = tags[tag_id]

            offset = ((page_im.size[0] // per_row) * i -
                      page_im.size[0] // (per_row * 2),
                      (page_im.size[1] // per_row) * j -
                      page_im.size[1] // (per_row * 2))  # To center the tag

            print(f'ID: {tag_id}')

            # Upscale the apriltag without interpolation
            img = im.resize((size_to_pixels, size_to_pixels),
                            Image.Resampling.NEAREST)  # Need the Image.NEAREST to remove blur
            page_im.paste(img, offset)  # centered by offset

            # create text
            text = f'ID: {tag_id}\nFamily: {tagFamily}\nSize: {resize_mm} mm'
            offset_text = (
                (page_im.size[0] // per_row) * i - page_im.size[0] // (per_row * 2),
                (page_im.size[1] // per_row) * j - page_im.size[1] // (
                        per_row * 4))  # To center the label
            draw.text(offset_text, text, (0, 0, 0), font=font)


def save(name: str, img: Image):
    heif_file = pillow_heif.from_pillow(page_im)
    heif_file.save(name, quality=-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', type=str, default='', help='type of transformations')
    parser.add_argument('--transform', type=str, default='./cfg/resize.yaml', help='type of transformations')

    opt = parser.parse_args()

    # A.transforms.register_transform(CustomTransform)

    # Dots per inch - Printer parameters
    dpi = 300
    # Image size disired in mm
    resize_mm = 6.5  # In millimetrs
    # Need a Apriltag with this values in folder
    tagFamily = 'tagCustom48h12'  # Must be same name that folder
    # tags quantity
    n_tags = 50
    # tags range
    tag_range_start = 300
    tag_range_end = 399
    # generate 3rd generation tags
    tg = TagGenerator3(tagFamily)

    # tags per row
    per_row = 5

    # size of the page
    page_size = (int(np.around((215.9 * dpi) / 25.4)), int(np.around((279.4 * dpi) / 25.4)))

    # Desired apriltag size to pixel based on dpi
    size_to_pixels = int(np.around((resize_mm * dpi) / 25.4))

    page_im = page(page_size)

    draw = ImageDraw.Draw(page_im)

    ids = random.sample(range(tag_range_start, tag_range_end), n_tags)

    if opt.kind:
        _transform = transform(opt.kind)
    else:
        _transform = A.load(opt.transform, data_format='yaml')

    # generate tags by ids
    tgs = gen_tags(transform=_transform, n_tags=n_tags, ids=ids)

    # draw tags on a page
    paint(tgs, per_row, page_im, draw)

    # save page to image file
    save(name='tags.heic', img=page_im)
