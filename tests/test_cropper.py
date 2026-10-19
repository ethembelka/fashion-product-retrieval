from PIL import Image

from fashret.data.cropper import expand_box, resize_short_side


def test_expand_box_adds_margin_and_clips():
    # 100x100 box at (10,10) with 10% margin -> grows by 10 px each side
    assert expand_box(10, 10, 110, 110, 0.10, 500, 500) == (0, 0, 120, 120)
    # clipped at image border
    assert expand_box(0, 0, 100, 100, 0.10, 105, 105) == (0, 0, 105, 105)


def test_resize_short_side_keeps_aspect_and_never_upsamples():
    img = Image.new("RGB", (400, 800))
    out = resize_short_side(img, 256)
    assert out.size == (256, 512)
    small = Image.new("RGB", (100, 200))
    assert resize_short_side(small, 256).size == (100, 200)
