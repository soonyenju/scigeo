import numpy as np
import pandas as pd

bands_table = pd.DataFrame.from_dict(
    {
        'Bands': ['Red', 'NIR', 'Blue'],
        'MODIS': ['B1', 'B2', 'B3'],
        'Landsat-7': ['B3', 'B4', 'B1'],
        'Landsat-8': ['B4', 'B5', 'B2'],
        'Sentinel-2': ['B4', 'B8', 'B2'],

    }
).set_index('Bands')

def get_NDVI(r, nir):
    ndvi = (nir - r) / (nir + r)
    return ndvi

def get_NIRv(r_ndvi, nir, ndvi_in = True):
    if ndvi_in:
        ndvi = r_ndvi
        nirv = ndvi * nir
    else:
        r = r_ndvi
        nirv = (nir - r) / (nir + r) * nir
    return nirv

def get_kNDVI(ndvi):
    kndvi = np.tanh(ndvi**2)
    return kndvi

def get_EVI2band(r, nir):
    evi = 2.5 * (nir - r) / (nir + 2.4 * r + 1)
    return evi

def get_EVI3band(r, nir, b):
    # EVI = G * ((NIR - R) / (NIR + C1 * R - C2 * B + L))
    evi = 2.5 * ((nir - r) / (nir + 6 * r - 7.5 * b + 1))
    return evi

def get_NDWI(green, nir):
    ndwi = (green - nir) / (green + nir)
    return ndwi


def get_MODIS_IGBPcode(number_first = False):
    '''
    url: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1 
    '''
    MODIS_IGBP_codes = ['ENF', 'EBF', 'DNF', 'DBF', 'MF', 'CSH', 'OSH', 'WSA', 'SAV', 'GRA', 'WET', 'CRO', 'URB', 'CNV', 'SNO', 'BSV', 'WAB']
    MODIS_IGBP_dict = dict(zip(MODIS_IGBP_codes, np.arange(len(MODIS_IGBP_codes)) + 1))
    if number_first:
        return {v: k for k, v in esa_class.items()}
    else:
        return MODIS_IGBP_dict

def get_DynamicWorld_code(number_first = False):
    google_class = {
        0: 'water',
        1: 'trees',
        2: 'grass',
        3: 'flooded_vegetation',
        4: 'crops',
        5: 'shrub_and_scrub',
        6: 'built',
        7: 'bare',
        8: 'snow_and_ice'
    }
    if number_first:
        return google_class
    else:
        return {v: k for k, v in google_class.items()}


def get_ESAWorldCover_code(number_first = False):
    # https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v100

    esa_class = {
        10: 'trees',
        20: 'shrub_and_scrub',
        30: 'grass',
        40: 'crops',
        50: 'built',
        60: 'bare',
        70: 'snow_and_ice',
        80: 'water',
        90: 'flooded_vegetation',
        95: 'mangroves',
        100: 'moss_and_lichen'
    }
    if number_first:
        return esa_class
    else:
        return {v: k for k, v in esa_class.items()}

def convert_gCm2d1_PgCyr_025deg():
    '''
    Terrestrial ecosystem carbon flux unit conversion:
    from gC m-2 d-1 to Pg C yr-1 for spatial resolution of 0.25 deg
    '''
    coef = 365 * 0.25 * 0.25 * 1e5 * 1e5 / 1e15
    return coef

def deg2m(longitude, latitude, scale_lon, scale_lat):
    # deg x deg => m2
    # Length in km of 1° of latitude = always 111.32 km
    # Length in km of 1° of longitude = 40075 km * cos( latitude ) / 360
    _, lats = np.meshgrid(longitude, latitude)
    coef_mat = 40075 * np.cos(np.deg2rad(np.abs(lats))) / 360 * 111.32 * 1e3 * 1e3
    coef_mat = coef_mat * scale_lon * scale_lat
    return coef_mat

def ETM2OLI(data, band):
    # Convert ETM+ reflectance to OLI
    # See Google Earth Engine page: https://developers.google.com/earth-engine/tutorials/community/landsat-etm-to-oli-harmonization
    ETM_TO_OLI_COEFFS = {
        'B1': (0.8474, 0.0003),  # ETM+ B1 → OLI B2 ('blue')
        'B2': (0.8483, 0.0088),  # ETM+ B2 → OLI B3 ('green')
        'B3': (0.9047, 0.0061),  # ETM+ B3 → OLI B4 ('red')
        'B4': (0.8462, 0.0412),  # ETM+ B4 → OLI B5 ('nir')
        'B5': (0.8937, 0.0254),  # ETM+ B5 → OLI B6 ('swir1')
        'B7': (0.9071, 0.0172),  # ETM+ B7 → OLI B7 ('swir2')
    }

    a = ETM_TO_OLI_COEFFS[band][0]
    b = ETM_TO_OLI_COEFFS[band][1]
    return data * a + b
