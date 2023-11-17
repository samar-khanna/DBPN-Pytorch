from PIL import Image
from torch.utils.data import Dataset
import random
import torch
import torchvision as tv
import torchvision.transforms as transforms
import numpy as np
from shapely.wkt import loads as shape_loads



CATEGORIES = ["airport", "airport_hangar", "airport_terminal", "amusement_park",
              "aquaculture", "archaeological_site", "barn", "border_checkpoint",
              "burial_site", "car_dealership", "construction_site", "crop_field",
              "dam", "debris_or_rubble", "educational_institution", "electric_substation",
              "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
              "gas_station", "golf_course", "ground_transportation_station", "helipad",
              "hospital", "impoverished_settlement", "interchange", "lake_or_pond",
              "lighthouse", "military_facility", "multi-unit_residential",
              "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park",
              "parking_lot_or_garage", "place_of_worship", "police_station", "port",
              "prison", "race_track", "railway_bridge", "recreational_facility",
              "road_bridge", "runway", "shipyard", "shopping_mall",
              "single-unit_residential", "smokestack", "solar_farm", "space_facility",
              "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth",
              "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility",
              "wind_farm", "zoo"]


class SentinelNormalize:
    mean = [1370.19151926, 1184.3824625, 1120.77120066, 1136.26026392,
            1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
            1972.62420416, 582.72633433, 14.77112979, 1732.16362238, 1247.91870117]
    std = [633.15169573, 650.2842772, 712.12507725, 965.23119807,
           948.9819932, 1108.06650639, 1258.36394548, 1233.1492281,
           1364.38688993, 472.37967789, 14.3114637, 1310.36996126, 1087.6020813]
    """
    Normalization for Sentinel-2 imagery, inspired from
    https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L111
    """

    def __init__(self, mean=None, std=None, channel_specific=True):
        if mean is None:
            mean = self.mean
        if std is None:
            std = self.std

        self.mean = np.array(mean)
        self.std = np.array(std)
        if not channel_specific:
            self.mean = self.mean.mean()
            self.std = self.std.mean()

    def __call__(self, x, *args, **kwargs):
        min_value = self.mean - 2 * self.std
        max_value = self.mean + 2 * self.std
        img = (x - min_value) / (max_value - min_value) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class SentinelFlipBGR:
    """
    Must call after conversion to Tensor
    """
    def __init__(self):
        pass

    def __call__(self, x, *args, **kwargs):
        x[1:4, :, :] = x[[3,2,1], :, :]
        return x


def get_patch(img_in, img_tar, img_bic, patch_size, scale, ix=-1, iy=-1):
    _, ih, iw = img_in.shape
    th, tw = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    # img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    img_in = transforms.functional.crop(img_in, iy, ix, ip, ip)
    # img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))
    img_tar = transforms.functional.crop(img_tar, ty, tx, tp, tp)
    # img_bic = img_bic.crop((ty, tx, ty + tp, tx + tp))
    img_bic = transforms.functional.crop(img_bic, ty, tx, tp, tp)

    # info_patch = {
    #     'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, img_bic #, info_patch


def fmow_preprocess_train(examples, patch_size=None, lowres=64, highres=512, is_train=True):
    low_res_transforms = transforms.Compose(
        [
            SentinelNormalize(channel_specific=True),
            transforms.ToTensor(),
            SentinelFlipBGR(),
            transforms.Resize(lowres, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(lowres),  # cond img can be 64x64, 128x128, 256x256, or 512x512
            # transforms.ToTensor(),
        ]
    )
    target_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(highres, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(highres),
        ]
    )

    for example in examples:
        upscale = highres // lowres
        input = low_res_transforms(example['multispectral.npy'])
        target = target_transforms(example['rgb.npy'])
        bicubic = transforms.functional.resize(
            transforms.functional.resize(target, lowres, interpolation=transforms.InterpolationMode.BICUBIC),
            highres, interpolation=transforms.InterpolationMode.BICUBIC
        )
        # bicubic = transforms.functional.resize(input, highres, interpolation=transforms.InterpolationMode.BICUBIC)

        if is_train and patch_size is not None:
            input, target, bicubic = get_patch(input, target, bicubic, patch_size, upscale)

        yield input, target, bicubic