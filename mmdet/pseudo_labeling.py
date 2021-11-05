from make_crop import make_crop_image
from mixed_image import make_miximage

if __name__ == "__main__":
    make_crop_image('swin_L384')
    make_miximage()
    print('complete')