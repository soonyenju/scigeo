# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 14:45:24 2020

@author: sz394@exeter.ac.uk
"""

import numpy as np
# ==================================================================== #
# Resource links                                                       #
# Gas variable conversion:                                             #
# https://www.gastec.co.jp/en/technology/knowledge/concentration/      #
# Something about water vapor:                                         #
# https://www.cnblogs.com/ruo-li-suo-yi/p/7772777.html                 #
# Online converter to check algorithm:                                 #
# https://www.cactus2000.de/uk/unit/masshum.shtml                      #
# engineering toolbox                                                  #
# https://www.engineeringtoolbox.com/                                  #
# ==================================================================== #

# -------------------------------------------------------------------
# Constants
R = universal_gas_constant = 8.314 # J/K/mol
Md = dry_air_molecular_weight = 28.96546e-3 # unit: kg / mol
MH2O = Mw = water_molecular_weight = 18.015268 # unit: g / mol
MCO2 = carbon_dioxide_molecular_weight =  44.009 # unit: g/mol
NA = 6.0221409e23 # Avogadro number
# -------------------------------------------------------------------
# Basic equations
def percent2ppm(percent):
    """
    math: ppm=%×10000 → 1ppm=0.0001%
    """
    ppm = percent * 10000
    return ppm

def ppm2percent(ppm):
    """
    math: %=ppm×1/10000 → 1%=10000ppm
    """
    percent = ppm / 10000
    return percent
def ppm2mass_density(ppm, M, T, P = 1013.25):
    """
    ppm → mg/m3
    Input:
    ------
    ppm(parts per million):
        This unit that expresses concentration in parts per million is measured as the volume (denoted in litres [L]) of a substance found in 1L of a medium such as air.
    M(g/mol): 
        Molecular weight of a substance
    T(degC):
        Temperature
    P(hPa): 
        P denotes the atmospheric pressure at the point of measurement (hPa)
    Output:
    -------
    mass density (mg/m3)
        This unit expresses the concentration in one cubic metre of air (equivalent to 1L or 1000mL) of a substance in terms of its mass (measured in milligrams). It is primarily used for particle-like substances, and only rarely for gaseous concentrations.
    Note:
    -----
    22.4(L): 
        The volume of 1 mol at 1 atmospheric pressure at 0 degC
    273(K): 
        FK stands for Kelvin, the unit used to measure thermodynamic temperature; as 0 degC corresponds to 273.15K.
        You simply need to add 273 to the Celsius/Centigrade value (273+T) to obtain the temperature in Kelvin
    1013(hPa): 
        One atmospheric pressure    
    math: 
    -----
    mg/m3=ppm×(M/22.4)×(273/(273+T))×(P/1013)
    """
    mass_density = ppm * (M / 22.4)  * (273.15 / (273.15 + T)) * (P / 1013.25)
    return mass_density

def mass_density2ppm(mass_density, M, T, P = 1013.25):
    """
    mg/m3 → ppm
    Input:
    ------
    mass_density (mg/m3)
        This unit expresses the concentration in one cubic metre of air (equivalent to 1L or 1000mL) of a substance in terms of its mass (measured in milligrams). It is primarily used for particle-like substances, and only rarely for gaseous concentrations.
    M(g/mol): 
        Molecular weight of a substance
    T(degC):
        Temperature
    P(hPa): 
        P denotes the atmospheric pressure at the point of measurement (hPa)
    Output:
    -------
    ppm(parts per million):
        This unit that expresses concentration in parts per million is measured as the volume (denoted in litres [L]) of a substance found in 1L of a medium such as air.
    Note:
    -----
    22.4(L): 
        The volume of 1 mol at 1 atmospheric pressure at 0 degC
    273(K): 
        FK stands for Kelvin, the unit used to measure thermodynamic temperature; as 0 degC corresponds to 273.15K.
        You simply need to add 273 to the Celsius/Centigrade value (273+T) to obtain the temperature in Kelvin
    1013(hPa): 
        One atmospheric pressure    
    math: 
    -----
    ppm=mg/m3×(22.4/M)×((273+T)/273)×(1013/P)
    mg/m3=ppm×(M/22.4)×(273/(273+T))×(P/1013)
    """
    ppm = mass_density * (22.4 / M) * ((273.15 + T) / 273.15 ) * (1013.25 / P)
    return ppm

def volumn_density2percent(volumn_density, M, T, P = 1013.25):
    """
    mg/L → %
    Input:
    ------
    volumn_density(mg/L (milligram per litre)):
        This unit expresses the concentration in one litre of air (1000mL) of a substance in terms of its mass (measured in milligrams). It is generally used for measuring concentrations in liquids, and only rarely for gaseous concentrations.
    M(g/mol): 
        Molecular weight of a substance 
    T(degC):
        Temperature
    P(hPa): 
        P denotes the atmospheric pressure at the point of measurement (hPa)
    Output:
    -------
    % (percent):
        This unit expresses concentration in parts per hundred (percentage) of a substance in 100mL of a medium such as air.
    math:
    -----
    %=mg/L×(22.4/M)×((273+T)/273)×(1/10)×(1013/P)
    """
    percent = volumn_density * (22.4 / M) * ((273.15 + T) / 273.15 ) * (1 / 10) * (1013.25 / P)
    return percent

def percent2volumn_density(percent, M, T, P = 1013.25):
    """
    % → mg/L
    Input:
    ------
    % (percent):
        This unit expresses concentration in parts per hundred (percentage) of a substance in 100mL of a medium such as air.
    M(g/mol): 
        Molecular weight of a substance 
    T(degC):
        Temperature
    P(hPa): 
        P denotes the atmospheric pressure at the point of measurement (hPa)
    Output:
    -------
    volumn_density(mg/L (milligram per litre)):
        This unit expresses the concentration in one litre of air (1000mL) of a substance in terms of its mass (measured in milligrams). It is generally used for measuring concentrations in liquids, and only rarely for gaseous concentrations.
    math:
    -----
    mg/L=%×(M/22.4)×(273/(273+T))×10×(P/1013)
    """
    volumn_density = percent * (M/ 22.4) * (273.15 / (273.15 + T)) * 10 * (P / 1013.25)
    return volumn_density

def saturation_vapor_pressure(T, es0 = 6.112):
    """
    Input:
    ------
    T(degC):
        Temperature
    es0 (hPa):
        saturation_vapor_pressure at 0 degC
    Output:
    -------
    saturation_vapor_pressure(hPa)
    math:
    -----
    Tetens formula
    6.112 e^((17.67 * T) / (T + 243.5)
    """
    es = 6.112 * np.e ** ((17.67 * T) / (T + 243.5))
    return es

def relative_humidity2dewpoint(T, RH):
    """
    Input:
    ------
    T(degC):
        Temperature
    RH:
        Relative humidity in the range [0 - 1]
    Output:
    -------
    Td(degC):
        Dewpoint temperature
    math:
    -----
    Magnus-Tetens Approximation
    Valid when:
        0 < T < 60
        1% < RH < 100%
        0 degC < Td < 50 degC
    """
    a = 17.27 # constant, unitless
    b = 237.7 # constant, degC
    gamma = (a * T) / (b + T) + np.log(RH) # np.log is ln
    Td = (b * gamma) / (a - gamma)
    return Td

def vapor_pressure2absolute_humidity(e, T):
    """
    Input:
    ------
    e(hPa): 
        vapor pressure
    T(degC):
        temperature
    Output:
    -------
    absolute_humidity(g/m3)
    Note:
    -----
    Rv: water vapor gas constant 461.5(J Kg^-1 K^-1)
    """
    Rv = 461.5
    # convert T from degC to Kelvin
    # convert e from hPa to Pa
    absolute_humidity = (e * 100)/(Rv * (T + 273.15)) # Unit: Kg/m3
    # Kg/m3 -> g/m3
    absolute_humidity = 1000 * absolute_humidity
    return absolute_humidity 

 
def specific_humidity(e, P):
    """
    Input:
    ------
    e(hPa): 
        vapor pressure
    P(hPa):
        atmospheric pressure
    Output:
    -------
    q(g/kg):
        Specific humidity
    math:
    -----
    q = mv / (mv + md) = mv/m = rhov / rho = (622 * e) / (p - 0.378 * e) g/Kg = (0.622 * e) / (P - 0.378 * e) g/g
    where
        mv: mass of vapor
        md: mass of dry air
        m: mass of wet air
        rhov: mass density of vapor
        rho: mass density of wet air
        P: atmospheric pressure (hPa)
    """
    q = (622 * e) / (P - 0.378 * e)   
    return q

def specific_humidity2vapor_pressure(q, P):
    # Added 27/09/2022
    # Example: specific_humidity2vapor_pressure(6.06, 1013.25) => 9.83563457971213
    """
    Input:
    ------
    q(g/kg):
        Specific humidity
    P(hPa):
        atmospheric pressure
    Output:
    -------
    e(hPa): 
        vapor pressure
    math:
    -----
    q = mv / (mv + md) = mv/m = rhov / rho = (622 * e) / (p - 0.378 * e) g/Kg = (0.622 * e) / (P - 0.378 * e) g/g
    e = P / (622/q + 0.378)
    where
        mv: mass of vapor
        md: mass of dry air
        m: mass of wet air
        rhov: mass density of vapor
        rho: mass density of wet air
        P: atmospheric pressure (hPa)
    """
    e = P / (622/q + 0.378) 
    return e

def mixing_ratio(e, P):
    """
    Input:
    ------
    e(hPa): 
        vapor pressure
    P(hPa):
        atmospheric pressure
    Output:
    -------
    r(g/kg):
        mixing ratio, the ratio of vapor mass to dry air mass
    math:
    -----
    r = mv / md = rhov / rhod = (e / (Rv * T)) / (Pd / (Rd * T)) = (Rd * e) / (Rv * (P - e)) 
      = 622 * e / (p - e) g/Kg = 0.622 * e / (p - e) g/g
    where
        mv: mass of vapor
        md: mass of dry air
        rhov: mass density of vapor
        rhod: mass density of dry air
        Rv: water vapor gas constant 461.5(J Kg^-1 K^-1)
        Rd: dry air gas constant 286.9(J Kg^-1 K^-1)
    """
    r = (622 * e) / (P - e)
    return r

def virtual_temperature(T, solution, r = None, q = None, Td = None, P = None):
    """
    Input:
    ------
    T(degC):
        temperature
    r(kg/kg): optional
        mixing ratio, the ratio of vapor mass to dry air mass
    q(kg/kg): optional
        Specific humidity
    Td(degC): optional
        Dewpoint temperature
    P(hPa): optional
        P denotes the atmospheric pressure at the point of measurement (hPa)
    Output:
    Tv(degC):
        virtual temperature
    ------        
    Note:
    -----
    Solution 1: https://glossary.ametsoc.org/wiki/Virtual_temperature#:~:text=(Also%20called%20density%20temperature.),and%20water%20vapor%2C%20%E2%89%88%200.622.
        use T and r
    Solution 2:
        use T and q
    Solution 3: https://www.weather.gov/media/epz/wxcalc/virtualTemperature.pdf
        use T, Td, and P
    """
    if solution == 1:
        assert r != None, "mixing ratio r(Kg/Kg) must be given for solution 1" 
        Tv = T * (1 + r / 0.622) / (1 + r)
    elif solution == 2:
        assert q != None, "specific humidity q(Kg/Kg) must be given for solution 2" 
        Tv=(1 + 0.608 * q)* T
    elif solution == 3:
        assert (Td != None) and (P != None), "dewpoint temperature Td(degC) and station pressure P(hPa) must be given for solution 3" 
        Tv = (T + 273.15) / (1 - 0.379 * (6.11 * 10**((7.5 * Td) / (237.7 + Td)) / P))
        Tv = Tv - 273.15      
    else:
        raise(Exception("solution must be given as one of 1, 2, and 3"))
        
    return Tv

# dry air density
def dry_air_density(T, P, solution = 1):
    """
    Parameters:
    -----------
    T(degC): 
        Air temperature
    P(hPa): 
        Pressure
    solution: int
        Choice of formula
    Returns:
    --------
    rho_dry(g/m3): 
        Dry air density
    math:
    -----
    rho = M/V
        where M is mass (kg), V is volume (m3)
    according to gas equation:
    rho = rho0 * (T0 * P) / (P0 * T) = 3.48
        where subscript 0 is stadard case, and P0 = 101.325 KPa, T0 = 273.15 K, rho0 (dry air density) is 1.293 kg/m3
    """
    # solution 1:
    if solution == 1:
        rho0 = 1.293e3
        rho_dry = rho0 * 273.15 / (T + 273.15) * P / 1013.25
    # solution 2:
    else: 
        P = P / 10 # unit: KPa
        rho_dry = 3.48 * P / (T + 273.15) # Kg/m3
        rho_dry = rho_dry * 1000
    return rho_dry

def wet_air_density(T, P, es, MH2O = 1.8015268e-2, Md = 28.96546e-3):
    """
    Parameters:
    -----------
    T(K): 
        Air temperature
    P(Pa):
        Pressure
    es(Pa):
        Saturation_vapor_pressure
    MH2O(kg/mol): optional
        Water molecular weight
    Md(kg/mol): optional
        Dry air molecular weight
    Returns:
    --------
    rho_wet(g/m3): wet air density
    theory:
    -------
    Ideal gas law and Dalton partial pressure
    This method has large bias.
    """
    rho_wet = (MH2O * es + (P - es) * Md) / (8.314 * T) # Kg/m3
    rho_wet = rho_wet * 1000
    return rho_wet    

def wet_air_density2(T, P, RH, es):
    """
    Parameters:
    -----------
    T(degC): 
        Air temperature
    P(hPa):
        Pressure
    RH: 
        Relative humidity in the range [0 - 1]
    es(hPa):
        Saturation_vapor_pressure
    Returns:
    --------
    rho_wet(g/m3): wet air density
    theory:
    -------
    Ideal gas law and Dalton partial pressure
    This method has large bias.
    """
    P = P / 10 # unit: KPa
    rho_wet = 3.48 * P/(T + 273.15) * (1- 0.378 * RH * es / P) # Kg/m3
    rho_wet = rho_wet * 1000
    return rho_wet

# -------------------------------------------------------------------
# Advanced equations
def vapor_pressure(RH, T):
    """
    Vapor pressure at temperature T is the saturation vapor pressure at the corresponding dewpoint temperature
    Input:
    ------
    RH:
        Relative humidity in the range [0 - 1]
    T(degC):
        temperature
    Output:
    -------
    e(hPa): vapor pressure at temperature T
    """
    Td = relative_humidity2dewpoint(T, RH)
    e = saturation_vapor_pressure(Td)
    return e 

def temperature_to_vpd(Td, T):
    """
    2023-09-29
    Vapour pressure deficit at dew temperature Td and air temperature T
    Input:
    ------
    Td(degC):
        dew temperature
    T(degC):
        temperature
    Output:
    -------
    vpd(hPa): vapor pressure deficit
    """
    vpd = saturation_vapor_pressure(T) - saturation_vapor_pressure(Td)
    return vpd

# -------------------------------------------------------------------
# Eddy Covariance meteo equations
def virtual_temperature_from_absolute_real_temperature(T, e, P):
    """
    Input:
    ------
    T(K):
        absolute real temperature
    e(hPa): 
        vapor pressure
    P(hPa):
        atmospheric pressure
    Output:
    -------
    virtual temperature(K)
    """
    virtual_temperature = T * (1 + 0.38 * e / P)
    return virtual_temperature

def absolute_real_temperature_from_virtual_temperature(T, e, P):
    """
    Input:
    ------
    T(K):
        virtual temperature
    e(hPa): 
        vapor pressure
    P(hPa):
        atmospheric pressure
    Output:
    -------
    absolute real temperature(K)
    """
    absolute_real_temperature = T / (1 + 0.38 * e / P)
    return absolute_real_temperature

def sonic_temperature_from_absolute_real_temperature(T, e, P):
    """
    Input:
    ------
    T(K):
        absolute real temperature
    e(hPa): 
        vapor pressure
    P(hPa):
        atmospheric pressure
    Output:
    -------
    sonic temperature(K)
    """
    sonic_temperature = T * (1 + 0.32 * e / P)
    return sonic_temperature

def absolute_real_temperature_from_sonic_temperature(T, e, P):
    """
    Input:
    ------
    T(K):
        absolute real temperature
    e(hPa): 
        vapor pressure
    P(hPa):
        atmospheric pressure
    Output:
    -------
    sonic temperature(K)    
    """
    absolute_real_temperature = T / (1 + 0.32 * e / P)
    return absolute_real_temperature