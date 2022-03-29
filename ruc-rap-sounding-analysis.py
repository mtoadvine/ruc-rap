###########################################################
#														  #
# 				RUC-RAP Sounding Analysis                 #
#                        Python3				          #
#				=========================				  #
#														  #
# 				 Author: Matthew Toadvine				  #
# 				 Email: mtoadvin@uncc.edu				  #
#				  Date: 20 December 2021				  #
#														  #
#	Purpose: This script reads in RUC-RAP Model data	  #
#			 in GRIB/GRIB2 format, produces vertical	  #
#			 sounding profiles and calculates common 	  #
#			 convective parameters. SHARPpy is the 		  #
#			 Pythonic method for the interpolation 		  #
#			 and calculations.							  #
#														  #
#  ------- Working Libraries as of March 5, 2022 -------  #
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

## Function to interpolate the parcel profile and calculate a variety of
## common convective parameters. All arguments MUST be passed as arrays
## of equal size (pressure, heights, temperature, dew point, wind direction,
## wind speed) which are extracted from the GRIB file
def sharppy_calc(p, h, t, td, windDir, windSpd):
    
    # Create an interpolated environmental profile using the passed arrays
    prof = profile.create_profile(profile='default', pres=p, hght=h, tmpc=t, dwpc=td, wdir=windDir, wspd=windSpd, missing=-9999, strictQC=False)

    # Create parcel profiles based on surface, most-unstable, and 100mb mean layer
    sfcpcl = params.parcelx(prof, flag=1) # Surface Parcel
    mupcl = params.parcelx(prof, flag=3)  # Most-Unstable Parcel
    mlpcl = params.parcelx(prof, flag=4) # 100 mb Mean Layer Parcel
    
    # Extract surface pressure and interpolate pressures at a variety of 
    # heights (500m, 1000m, 2000m, 3000m, 4000m, 5000m, 6000m)
    sfc = prof.pres[prof.sfc]
    p500m = interp.pres(prof, interp.to_msl(prof, 500.))
    p1km = interp.pres(prof, interp.to_msl(prof, 1000.))
    p2km = interp.pres(prof, interp.to_msl(prof, 2000.))
    p3km = interp.pres(prof, interp.to_msl(prof, 3000.))
    p4km = interp.pres(prof, interp.to_msl(prof, 4000.))
    p6km = interp.pres(prof, interp.to_msl(prof, 6000.))

    # Calculate LCL, LFC, EL heights (m) and pressure levels (mb)
    lcl = sfcpcl.lclhght
    lcl_p = sfcpcl.lclpres
    
    lfc = sfcpcl.lfchght
    lfc_p = sfcpcl.lfcpres
    
    el = sfcpcl.elhght
    el_p = sfcpcl.elpres

    # Calculate all CAPE and CIN values for SB, ML, MU, and 0-3km (J/kg)
    sbCape = sfcpcl.bplus
    sbCin = sfcpcl.bminus
    sbCape03 = sfcpcl.b3km
    mlCape = mlpcl.bplus
    mlCin = mlpcl.bminus
    mlCape03 = mlpcl.b3km
    muCape = mupcl.bplus
    muCin = mupcl.bminus

    # Calculate lapse rates for 0-3km and 700-500hPa layers
    lr03 = params.lapse_rate(prof, 0, 3000., pres=False)
    lr700500 = params.lapse_rate(prof, 700, 500, pres=True)
    
    # Calculate Theta-E for SB and ML parcels
    sbThE = thermo.ctok(thermo.thetae(sfcpcl.pres, sfcpcl.tmpc, sfcpcl.dwpc))
    mlThE = thermo.ctok(thermo.thetae(mlpcl.pres, mlpcl.tmpc, mlpcl.dwpc))
    
    # Calculate relative humidity layers (TCTOR specific, you may not need these)
    mlRh = params.mean_relh(prof, pbot=sfc, ptop=sfc-100)
    rh02 = params.mean_relh(prof, pbot=sfc, ptop=p2km)
    rh24 = params.mean_relh(prof, pbot=p2km, ptop=p4km)
    rh46 = params.mean_relh(prof, pbot=p4km, ptop=p6km)
    
    #Calculate DCAPE
    dCape = params.dcape(prof)

    # Calculate the effective inflow layer and print out the heights of the top
    # and bottom of the layer.  We'll have to convert it from m MSL to m AGL.
    eff_inflow = params.effective_inflow_layer(prof,mupcl=mupcl)
    ebot_hght = interp.to_agl(prof, interp.hght(prof, eff_inflow[0]))
    etop_hght = interp.to_agl(prof, interp.hght(prof, eff_inflow[1]))
    effDepth = etop_hght - ebot_hght
    hlfStmDep = (mupcl.elhght - ebot_hght) * 0.5
    
    # Calculate the 0-1, 0-3, and 0-6 km wind shear
    sfc_1km_shear = winds.wind_shear(prof, pbot=sfc, ptop=p1km)
    sfc_3km_shear = winds.wind_shear(prof, pbot=sfc, ptop=p3km)
    sfc_6km_shear = winds.wind_shear(prof, pbot=sfc, ptop=p6km)
    shear_1km = utils.KTS2MS(utils.comp2vec(sfc_1km_shear[0], sfc_1km_shear[1])[1])
    shear_3km = utils.KTS2MS(utils.comp2vec(sfc_3km_shear[0], sfc_3km_shear[1])[1])
    shear_6km = utils.KTS2MS(utils.comp2vec(sfc_6km_shear[0], sfc_6km_shear[1])[1])
    
    # Calculate the 0-500m and 0-6km pressure-weighted mean winds for BRN Shear Calculation
    mean_500m = winds.mean_wind(prof, pbot=sfc, ptop=p500m)
    mean_6km = winds.mean_wind(prof, pbot=sfc, ptop=p6km)
    mag_500m = utils.KTS2MS(utils.mag(mean_500m[0], mean_500m[1]))
    mag_6km = utils.KTS2MS(utils.mag(mean_6km[0], mean_6km[1]))
    brnShear = 0.5 * ((mag_500m - mag_6km)**2)
    
    # Calculate the effective wind shear (Effective Bulk Wind Difference)
    ebwd = winds.wind_shear(prof, pbot=eff_inflow[0], ptop=eff_inflow[1])
    ebwspd = utils.KTS2MS(utils.mag(ebwd[0], ebwd[1]))

    # Calculate the Bunkers Storm Motion Left and Right mover vectors (these are returned in u,v space
    # so we transform them into wind speed and direction space.) Needed for SRH calculations.
    srwind = params.bunkers_storm_motion(prof)

    # Calculate Storm-relative Helicity values for the 0-500m, 0-1km, 0-3km, and effective inflow layers
    srh0500m = winds.helicity(prof, 0, 500., stu = srwind[0], stv = srwind[1])
    srh1km = winds.helicity(prof, 0, 1000., stu = srwind[0], stv = srwind[1])
    srh3km = winds.helicity(prof, 0, 3000., stu = srwind[0], stv = srwind[1])
    effective_srh = winds.helicity(prof, ebot_hght, etop_hght, stu = srwind[0], stv = srwind[1])

    # Composite indices (e.g. STP, SCP) can be calculated after determining the effective inflow layer.
    # Fixed-Layer SCP derived from Thompson et al. (2003)
    scp_fix = (mupcl.bplus / 1000.) * (srh3km[0] / 100.) * (brnShear / 40.)
    scp_eff = params.scp(mupcl.bplus, effective_srh[0], ebwspd)
    stp_fix = (mlpcl.bplus / 1000.) * (shear_6km / 20.) * (srh1km[0] / 100.) * ((2000. - mlpcl.lclhght) / 1500.)
    stp_eff = params.stp_cin(mlpcl.bplus, effective_srh[0], ebwspd, mlpcl.lclhght, mlpcl.bminus)
    
    # Calculate fixed-layer (0-3km) and effective-layer SHERB (sherb3 and sherbe, respectively)
    # Both parameters derived from Sherburn and Parker (2014), effective-layer is manually calculated
    # because of modules not playing nice. Bummer.
    sherbs3 = params.sherb(prof, effective=False, ebottom=sfc, etop=p3km)
    sherbe = (ebwspd / 27.) * (lr03 / 5.2) * (lr700500 / 5.6)
    
    # Return all of the calculated parameters back to the main program (a long list, I know)
    return lcl, lfc, el, sbCape, sbCin, sbCape03, mlCape, mlCin, mlCape03, muCape, muCin, lr03, lr700500, sbThE, mlThE, mlRh, rh02, rh24, rh46, dCape, ebot_hght, etop_hght, effDepth, hlfStmDep, shear_1km, shear_3km, shear_6km, brnShear, ebwspd, srh0500m, srh1km, srh3km, effective_srh, scp_fix, scp_eff, stp_fix, stp_eff, sherbs3, sherbe


###########################################################
#	PROCESS GRIB DATA
###########################################################

## Define the files that we will be working with
wrkDir = '/home/mtoadvine/METR4105/'
grbFile = wrkDir + 'Python/rap_130_20200426_0200_000.grb2'
      
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
#td = t - ((100. - (rh*100.))/5.)


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
#	CALCULATE SOUNDING PARAMETERS (FINALLY)
###########################################################

## A long list of returned parameters, but this function (that we defined above)
## calculates all of these for us
lcl, lfc, el, sbCape, sbCin, sbCape03, mlCape, mlCin, mlCape03, muCape, muCin, lr03, lr700500, sbThE, mlThE, mlRh, rh02, rh24, rh46, dCape, ebot_hght, etop_hght, effDepth, hlfStmDep, shear_1km, shear_3km, shear_6km, brnShear, ebwspd, srh0500m, srh1km, srh3km, effective_srh, scp_fix, scp_eff, stp_fix, stp_eff, sherbs3, sherbe = sharppy_calc(pres,h,t,td,windDir,speed)


###########################################################
#	PLOT THE SKEW-T
###########################################################

## Change default size to be better for skew-T
fig = plt.figure(figsize=(12,10))
skew = SkewT(fig,rotation=45)

## Plot the data using normal plotting functions, in this case using
## log scaling in Y, as dictated by the typical meteorological plot
skew.plot(pres, t, 'r')
skew.plot(pres, td, 'g')

# Play around with these to determine how dense you want your wind barbs to be
# on your plots. This plots all wind barbs at all available pressure levels.
# Given that there's only 38-ish levels, you might not need to change this.
skew.plot_barbs(pres[:], u[:], v[:])

## Add the relevant special lines that make the Skew-T readable/understandable
skew.plot_dry_adiabats(t0=np.arange(233,533,10)*units('K'),alpha=0.25)
t0=np.arange(233,323,5)
skew.plot_moist_adiabats(t0=np.arange(233,323,5)*units('K'),alpha=0.25)
skew.plot_mixing_lines(alpha=0.25)
skew.ax.set_ylim(1050, 100)
skew.ax.set_xlim(-30,40)


## Add Surface-based parcel profile to plot as black line
## Requires that the variables have associated units
prof = mpcalc.parcel_profile(pres, t[0], td[0])
skew.plot(pres, prof, 'k', linewidth=2)

## Add CAPE/CIN shading - comment out if you don't want these
skew.shade_cin(pres,t,prof,td)
skew.shade_cape(pres,t,prof)

## Make some titles
plt.title('My Beautiful Sounding') 

## This makes your hodograph. It's in a weird place so that it doesn't impact
## the title location (it happened to me)
ax_hod = inset_axes(skew.ax, '25%', '25%', loc=1)
ho = Hodograph(ax_hod, component_range=40.)
ho.add_grid(increment=20)
ho.plot_colormapped(u, v, h)  # Plot a line colored by height

## Plot Sounding Parameters as text on the figure
textstr = 'LCL: ' +str(np.around(lcl)) + ' m\nLFC: '+ str(np.around(lfc)) + ' m\nEL: ' + str(np.around(el)) + ' m\n\nSBCAPE: ' + str(np.around(sbCape,1)) + ' J/kg\nSBCIN: ' + str(np.around(sbCin,1)) + ' J/kg\nSBCAPE03: ' + str(np.around(sbCape03,1)) + ' J/kg\nMLCAPE: ' + str(np.around(mlCape,1)) + ' J/kg\nMLCIN: ' + str(np.around(mlCin,1)) + ' J/kg\nMLCAPE03: ' + str(np.around(mlCape03,1)) + ' J/kg\nMUCAPE: ' + str(np.around(muCape,1)) + ' J/kg\nMUCIN: ' + str(np.around(muCin,1)) + ' J/kg\nLR03: ' + str(np.around(lr03,2)) + ' C/km\nLR700500: ' + str(np.around(lr700500,2)) + ' C/km\n\nSBTHE: ' + str(np.around(sbThE,1)) + ' K\nMLTHE: ' + str(np.around(mlThE,1)) + ' K\nMLRH: ' + str(np.around(mlRh,1)) + ' %\nRH02: ' + str(np.around(rh02,1)) + ' %\nRH24: ' + str(np.around(rh24,1)) + ' %\nRH46: ' + str(np.around(rh46,1)) + ' %\nDCAPE: ' + str(np.around(dCape[0],1)) + ' J/kg\n\nEFFBOT: ' + str(np.around(ebot_hght,1)) + ' m\nEFFTOP: ' + str(np.around(etop_hght,1)) + ' m\nEFFDEP: ' + str(np.around(effDepth,1)) + ' m\nSTMHLF: ' + str(np.around(hlfStmDep,1)) + ' m\n\nSHEAR01: ' + str(np.around(shear_1km,2)) + ' m/s\nSHEAR03: ' + str(np.around(shear_3km,2)) + ' m/s\nSHEAR06: ' + str(np.around(shear_6km,2)) + ' m/s\nBRNSHEAR: ' + str(np.around(brnShear,2)) + ' m2/s2\nEFFSHEAR: ' + str(np.around(ebwspd,2)) + ' m/s\n\nSRH05: ' + str(np.around(srh0500m[0],1)) + ' m2/s2\nSRH01: ' + str(np.around(srh1km[0],1)) + ' m2/s2\nSRH03: ' + str(np.around(srh3km[0],1)) + ' m2/s2\nSRHEFF: ' + str(np.around(effective_srh[0],1)) + ' m2/s2\n\nSCPFIX: ' + str(np.around(scp_fix,2)) + '\nSCPEFF: ' + str(np.around(scp_eff,2)) + '\nSTPFIX: ' + str(np.around(stp_fix,2)) + '\nSTPEFF: ' + str(np.around(stp_eff,2)) + '\nSHERBS3: ' + str(np.around(sherbs3,2)) + '\nSHERBE: ' + str(np.around(sherbe,2))
plt.text(70, -310, textstr)

## Save/show the plot
plt.savefig('mySounding.png',dpi=300,bbox_inches='tight')
plt.show()
        
## Print out the timing just for our own interest
print(scriptStart)
print("Script End: " + strftime('%Y-%m-%d %H:%M:%S'))
