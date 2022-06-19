###########################################################
#														  #
# 			 RUC-RAP to CM1 Initial Sounding              #
#                        Python3				          #
#				=========================				  #
#														  #
# 				 Author: Matthew Toadvine				  #
# 				 Email: mtoadvin@uncc.edu				  #
#				  Date: 18 June 2022			     	  #
#														  #
#	Purpose: This script reads in RUC-RAP Model data	  #
#			 in GRIB/GRIB2 format, and produces     	  #
#			 CM1/WRF initial sounding text files.      	  #
#														  #
#  ------- Working Libraries as of June 18, 2022 -------  #
#                                                         #
#       Numpy           1.20.3                            #
#       MetPy           1.2                               #
#       MatPlotLib      3.5                               #
#       cfgrib          0.9.9.0                           #
#       SHARPpy         1.4                               #
#       Spyder IDE      5.1.5                             #
#														  #
###########################################################


###########################################################
#	IMPORT REQUIRED LIBRARIES/MODULES
###########################################################

# Numpy for array structure
import numpy as np

# MetPy for units and Skew-T/Hodograph plots
import metpy.calc as mpcalc
from metpy.units import units
from metpy.plots import SkewT,Hodograph

# Import all native math functions
# If you remove this module, you need to use Numpy math functions
from math import *

# Engine to process GRIB files
import cfgrib

# MatPlotLib for plotting figures
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# SHARPpy for profile interpolation and parameter calculations
import sharppy
import sharppy.sharptab.profile as profile
import sharppy.sharptab.interp as interp
import sharppy.sharptab.winds as winds
import sharppy.sharptab.utils as utils
import sharppy.sharptab.params as params
import sharppy.sharptab.thermo as thermo

# Time module to keep track of script runtime - helpful to have
from time import strftime

# This is just to silence warnings from SHARPpy, don't be scared
import warnings
warnings.filterwarnings("ignore")


###########################################################
#	BEGIN MAIN PROGRAM
###########################################################

# Just to help gauge how fast the script runs
scriptStart = "Script Start: " + strftime('%Y-%m-%d %H:%M:%S')
print(scriptStart)


###########################################################
#	DEFINE INTERPOLATION/PARAMETER FUNCTIONS
###########################################################

## Function to interpolate the parcel profile to AGL coordinates from
## original pressure coordinates. All arguments MUST be passed as arrays
## of equal size (pressure, heights, temperature, dew point, and 
## wind components) which are extracted from the GRIB file
def sharppy_interp(p, h, t, td, u, v):

    # Create our environment profile
    prof = profile.create_profile(profile='default', pres=p, hght=h, tmpc=t, dwpc=td, u=u, v=v, missing=-9999, strictQC=False)
    
    # Create an array for AGL height values using 10m vertical resolution
    agl_hght = []
    for i in range(0,18001,10):
        agl_hght.append(i)

    
    # Interpolate pressures based on AGL heights as long as pressures are less than 100hPa
    # This is because all other interpolation is based on pressure levels
    p = []
    h = []
    for i in range(len(agl_hght)):
        pressure = interp.pres(prof, interp.to_msl(prof, agl_hght[i]))
        if pressure >= 100.:
            p.append(pressure)
            h.append(agl_hght[i])

    
    # Interpolate T, Td, and wind vectors based on new pressure profile
    t = []
    td = []
    windDir = []
    windSpd = []
    for i in range(len(p)):
        t.append(interp.temp(prof,p[i]))
        td.append(interp.dwpt(prof,p[i]))
        windDir.append(np.take((interp.vec(prof,p[i])[0].data),0))
        windSpd.append(interp.vec(prof,p[i])[1])
    
    # Convert these new lists to Numpy arrays and assign units
    p = np.array(p) * units('hPa')
    h = np.array(h) * units('m')
    t = np.array(t) * units('degC')
    td = np.array(td) * units('degC')
    windDir = np.array(windDir) * units('degrees')
    windSpd = np.array(windSpd) * units('m/s')
    
    # Return all new arrays to main program
    return p, h, t, td, windDir, windSpd


###########################################################
#	PROCESS GRIB DATA
###########################################################

## Define the files that we will be working with
wrkDir = 'your/working/path/here/'
grbFile = wrkDir + 'your-file-name-here'
      
## This is where you grab the data from the file. This is a lot of trial and
## error depending on what you need. These should suffice for what you want to
## do. When extracted, the data is stored in the datasets and you can pull 
## individual variables from them. Use print(ds1) to see what variables are
## available in each dataset
ds1 = cfgrib.open_dataset(grbFile, filter_by_keys={'stepType': 'instant', 'typeOfLevel': 'isobaricInhPa'})
ds2 = cfgrib.open_dataset(grbFile, filter_by_keys={'typeOfLevel': 'heightAboveGround','level':2})
ds3 = cfgrib.open_dataset(grbFile, filter_by_keys={'stepType': 'instant', 'typeOfLevel': 'surface'})
ds4 = cfgrib.open_dataset(grbFile, filter_by_keys={'typeOfLevel': 'heightAboveGround','level':10})
            
## A selection of variables that are extracted from the saved datasets
# Isobaric variables
geoHeight = ds1.gh
isobarTemp = ds1.t
isobarRh = ds1.r
isobarU = ds1.u
isobarV = ds1.v
lat = ds1.latitude.values
lon = ds1.longitude.values
p = isobarTemp.isobaricInhPa # Isobaric pressure levels

# Surface variables (2m temp/dewpoint; 10m winds)
surfaceT = ds2.t2m
surfaceTd = ds2.d2m
surfaceP = ((ds3.sp) / 100.) # Originally in Pa, convert to hPa
surfaceU = ds4.u10
surfaceV = ds4.v10

## Define your latitude/longitude of interest
## Latitude format is as we normally use it
## Longitude ranges from 0-360 degrees: so -85 W = 295 (see comment below)
myLat = 35
myLon = -80.8  + 360.    # If your longitude is -XX.XX W, add 360 to it

## How you want to do this next part is up to you. You need to convert
## your lat/lon coordinates into the nearest index of the lat/lon arrays
## This is how I did it, finding the smallest difference between both lat/lon

# Define some relatively large lat/lon differences to start
latDiff = 10.
lonDiff = 10.

# Loop through each latitude (array column 0)
for i in range(len(lat)):
    
    # Loop through each longitude (array column 1)
    for j in range(len(lon[0])):
        
        # A simple conditional that takes the difference magnitude and checks
        # if it is smaller than the most recent difference value (latDiff/lonDiff).
        # This finds the nearest lat/lon indices (i,j) so you can pull a specific
        # column nearest to your point of interest.
        if (abs(lat[i][j] - myLat) < latDiff):
            latDiff = abs(lat[i][j] - myLat)
            myI = i
        if (abs(lon[i][j] - myLon) < lonDiff):
            lonDiff = abs(lon[i][j] - myLon)
            myJ = j
            
# If you need confirmation, here are some print statements to show the values.
# Because the lat/lon arrays are both 2-D of the same size, you'll use myLati
# and myLonj indices in both arrays as shown below
print(myLat,lat[myI,myJ])
print(myLon,lon[myI,myJ])


###########################################################
#	BUILD SOUNDING DATA ARRAYS
###########################################################

## This looks a bit chaotic, but it's all to ensure that the lowest level
## in the isobaric arrays does not equal the surface (not common, but it
## could happen). As long as they don't match, this block will insert the 
## surface data in the correct position when joining the surface and upper
## air data. I'll explain along the way.
    
## Isolate surface/2m/10m values at our specific lat/lon
sfcT = surfaceT.values[myI,myJ]
sfcTd = surfaceTd.values[myI,myJ]
sfcP = surfaceP.values[myI,myJ]
u10 = surfaceU.values[myI,myJ]
v10 = surfaceV.values[myI,myJ]

## This is what puts the surface level in the correct position
# First, create an empty list to contain our new vertical pressure profile
pres = []
# This is used later, just a boolean indicating if the surface pressure
# matches with the lowest pressure level
sfcPresMatch = False    

# Loop through each pressure level
for i in range(len(p)):
    
    # Start by checking if the surface pressure is greater than or equal
    # to our lowest isobaric level
    if i == 0:
        
        # If greater, append surface pressure first, then isobaric
        if sfcP > p[i].values:
            pres.append(sfcP)
            pres.append(p[i].values)
            
            # This surface index identifies which isobaric level is the lowest
            # for our particular location based on where the surface level is
            # Used next to pull only necessary levels from temp, RH, winds
            sfc_index = i
            
        # If equal, just use the isobaric level as the surface and change our
        # boolean to True
        elif sfcP == p[i].values:
            pres.append(p[i].values)
            sfc_index = i
            sfcPresMatch = True
    
    # If the surface pressure is less than the lowest isobaric level, where 
    # does it belong? Check to see if it is between layers and append the 
    # layers accordingly.
    else:
        if sfcP > p[i].values and sfcP < p[i-1].values:
            pres.append(sfcP)
            pres.append(p[i].values)
            sfc_index = i
        
        # If it no longer falls between a layer (already been used), simply
        # append the remaining isobaric layers
        elif sfcP > p[i].values:
            pres.append(p[i].values)


## Similar to above, we need to reconstruct the heights, temps, RH, and wind
## component arrays to match the new pressure levels that include the surface
# Create empty lists to hold the new data
heights = []
temp_values = []
rh_values = []
u = []
v = []

# Loop through the isobaric layers beginning with the previously identified
# lowest necessary layer and append the values at our lat/lon to the new lists
for i in range(sfc_index,len(isobarTemp)):    
    temp_values.append(isobarTemp[i].values[myI,myJ])
    rh_values.append(isobarRh[i].values[myI,myJ])
    u.append(isobarU[i].values[myI,myJ])
    v.append(isobarV[i].values[myI,myJ])
    heights.append(geoHeight[i].values[myI,myJ])

# Insert 10m u and v wind components at the 0 position of each list
# Yes, they're done twice. It's an assumption that the 2m and 10m winds
# are roughly equal, simply to make the final array match in size
if sfcPresMatch != True:
    u.insert(0,u10)
    v.insert(0,v10)
    u.insert(0,u10)
    v.insert(0,v10)

## Convert our lists into numpy arrays for MetPy use with appropriate units
t = np.array(temp_values)*units('kelvin')
rh = np.array(rh_values)*units('percent')
pres = np.array(pres)*units('hPa')
h = np.array(heights)*units('m')


## The following block is a lot of housekeeping. Conversions from RH to dewpoint,
## calculating wind speed and direction, interpolating the lowest two levels'
## temps, dewpoints, heights, pressures using linear interpolation. All 
## necessary before we calculate our sounding parameters

# Calculate wind speed in meters per second
speed = []
for i in range(len(u)):
    speed.append(np.sqrt((u[i]**2) + (v[i]**2)))

# Assign units to our wind component and speed arrays, and calculate
# wind direction using MetPy
u = np.array(u)*units('m/s')
v = np.array(v)*units('m/s')
speed = np.array(speed)*units('m/s')
windDir = 90. - np.arctan2(-v, -u)
windDir = windDir*units('degree')


# Calculate the dewpoint temperature using MetPy
td = mpcalc.dewpoint_from_relative_humidity(t,rh)


# Insert our surface (2m) values and 10m values to our final arrays if 
# surface pressure and lowest level are different
if sfcPresMatch != True:
    
    # Insert 2m temps
    t = np.insert(t,0,sfcT*units('kelvin'))
    
    # Interpolate and insert 10m temps
    t10m = (((t[1] - t[0]) / (h[1] - 2.0*units('m'))) * 10.0*units('m')) + t[0]
    t = np.insert(t,1,t10m)

    # Insert 2m Td
    td = np.insert(td,0,sfcTd*units('kelvin'))
    
    # Interpolate and insert 10m Td
    td10m = (((td[1] - td[0]) / (h[1] - 2.0*units('m'))) * 10.0*units('m')) + td[0]
    td = np.insert(td,1,td10m)

    # Interpolate 10m pressure
    p10m = (((pres[1] - pres[0]) / (h[1] - 2.0*units('m'))) * 10.0*units('m')) + pres[0]
    pres = np.insert(pres,1,p10m)

    # Add our 10m and 2m heights
    h = np.insert(h,0,10.0*units('m'))
    h = np.insert(h,0,2.0*units('m'))

## Bring all of our arrays back together in one place, ensuring that they are
## Numpy arrays with correct units
t = t.to('degC')            # Convert to deg C from Kelvin
td = td.to('degC')          # Convert to deg C from Kelvin
speed = speed.to('knots')   # Convert to knots from m/s



###########################################################
#	CONVERT SOUNDING TO AGL COORDINATES
###########################################################

## Pass arrays into the function that interpolates to AGL from pressure
## coordinates (allows for more high-resolution initial soundings)
p, h, t, td, windDir, windSpd = sharppy_interp(pres, h, t, td, u, v)

## Calculate our U, V wind components based on interpolated direction/speed
u, v = mpcalc.wind_components(windSpd, windDir)

## Dewpoint needs to be converted to mixing ratio ==> Td -> RH -> w
## The multiplication by 1000 at the end of the w calculation converts us
## from kg/kg --> g/kg which is required by CM1/WRF for initial soundings
rh = mpcalc.relative_humidity_from_dewpoint(t.to('K'), td.to('K'))
w = (mpcalc.mixing_ratio_from_relative_humidity(p.to('Pa'), t.to('K'), rh)) * 1000. 

## Temperature must be converted to Potential Temperature (Theta)
theta = mpcalc.potential_temperature(p.to('Pa'), t.to('K'))


###########################################################
#	WRITE OUT CM1/WRF SOUNDING
###########################################################

## Open the output sounding file
sndOut = open("CM1-initial_sounding.txt",'w')

## Write the first line (Surface Pressure (mb)-- Surface Theta (K) -- Surface Vapor Mixing Ratio (g/kg))
sndOut.write('{:>10.4f}\t{:>9.4f}\t{:>9.4f}\n'.format(np.around(p[0].data,4), 
                                                        np.around(theta[0].data,4), 
                                                        np.around(w[0].data,4)))

## Loop through each 100m heights and write out the height (m), temp (K), mixing ratio (g/kg), u and v (m/s)
for i in range(10,len(p),10):
    sndOut.write('{:>10.4f}\t{:>9.4f}\t{:>9.4f}\t{:>9.4f}\t{:>9.4f}\n'.format(np.around(h[i].data,4), 
                                                                                np.around(theta[i].data,4), 
                                                                                np.around(w[i].data,4), 
                                                                                np.around(u[i].data,4), 
                                                                                np.around(v[i].data,4)))
## Close the output file
sndOut.close()

        
## Print out the timing just for our own interest
print(scriptStart)
print("Script End: " + strftime('%Y-%m-%d %H:%M:%S'))
