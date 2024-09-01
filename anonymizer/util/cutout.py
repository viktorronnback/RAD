from PIL import Image


def _create_empty_transparent(width: int, height: int) -> Image.Image:
    """ Returns empty and transparent RGBA image """
    return Image.new("RGBA", (width, height), (0, 0, 0, 0))


def _gen_cutout_helper(original, background, mask):
    assert original.size == background.size == mask.size, \
        f"Image sizes must match! {original.size} == {background.size} == {mask.size}"
    
    original = original.convert("RGBA")
    background = background.convert("RGBA")

    mask = mask.convert("1") # 1 bit per pixel

    return Image.composite(image1=original, image2=background, mask=mask)


def gen_inv_cutout(original: Image.Image, mask: Image.Image) -> Image.Image:
    """ Returns inverted cutout (everything outside mask (black pixels) from original image) """
    transparent = _create_empty_transparent(original.width, original.height)

    return _gen_cutout_helper(original=transparent, background=original, mask=mask)


def gen_cutout_black_bg(original: Image.Image, mask: Image.Image) -> Image.Image:
    white_bg = Image.new("RGB", (original.width, original.height), (0,0,0))

    return _gen_cutout_helper(original=original, background=white_bg, mask=mask)


def gen_cutout_white_bg(original: Image.Image, mask: Image.Image) -> Image.Image:
    white_bg = Image.new("RGB", (original.width, original.height), (255,255,255))

    return _gen_cutout_helper(original=original, background=white_bg, mask=mask)


def gen_cutout(original: Image.Image, mask: Image.Image) -> Image.Image:
    """ Returns cutout (everything inside mask (white pixels) from original image) """
    transparent = _create_empty_transparent(original.width, original.height)

    return _gen_cutout_helper(original=original, background=transparent, mask=mask)
