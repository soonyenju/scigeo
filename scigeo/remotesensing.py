import numpy as np

def get_NDVI(r, nir):
    ndvi = (nir - r) / (nir + r)
    return ndvi

def get_NIRv(ndvi, nir):
    nirv = ndvi * nir
    return nirv

def get_kNDVI(ndvi):
    kndvi = np.tanh(ndvi**2)
    return kndvi

def get_EVI2band(r, nir):
    evi = 2.5 * (nir - r) / (nir + 2.4 * r + 1)
    return evi


def get_MODIS_IGBPcode():
    '''
    url: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1 
    '''
    MODIS_IGBP_codes = ['ENF', 'EBF', 'DNF', 'DBF', 'MF', 'CSH', 'OSH', 'WSA', 'SAV', 'GRA', 'WET', 'CRO', 'URB', 'CNV', 'SNO', 'BSV', 'WAB']
    MODIS_IGBP_dict = dict(zip(MODIS_IGBP_codes, np.arange(len(MODIS_IGBP_codes)) + 1))
    return MODIS_IGBP_dict

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



class VegIdx:
    def __init__(self):
      pass
    # NDVI
    # --------------------------------------------------------------------------------------------------------------
    @staticmethod
    def modis_ndvi(b1, b2):
        return (b2 - b1) / (b2 + b1)

    @staticmethod
    def landsat7_ndvi(b3, b4):
        return (b4 - b3) / (b4 + b3)

    @staticmethod
    def landsat8_ndvi(b4, b5):
        return (b5 - b4) / (b5 + b4)

    @staticmethod
    def sentinel2_ndvi(b8, b4):
        return (b8 - b4) / (b8 + b4)

    # EVI
    # --------------------------------------------------------------------------------------------------------------
    # EVI = G * ((NIR - R) / (NIR + C1 * R - C2 * B + L))
    @staticmethod
    def modis_evi_2band(b1, b2):
        return 2.5 * ((b2 - b1) / (b2 + 2.4 * b1 + 1))

    @staticmethod
    def modis_evi_3band(b1, b2, b3):
        return 2.5 * ((b2 - b1) / (b2 + 6 * b1 - 7.5 * b3 + 1))

    @staticmethod
    def landsat7_evi(b1, b3, b4):
        return 2.5 * ((b4 - b3) / (b4 + 6 * b3 - 7.5 * b1 + 1))

    @staticmethod
    def landsat8_evi(b2, b4, b5):
        return 2.5 * ((b5 - b4) / (b5 + 6 * b4 - 7.5 * b2 + 1))

    @staticmethod
    def sentinel2_evi(b2, b4, b8):
        return 2.5 * ((b8 - b4) / (b8 + 6 * b4 - 7.5 * b2 + 1))

