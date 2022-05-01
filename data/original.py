#!/usr/bin/env python
# coding: utf-8

# # Descriptive analysis of oceanographic data
# -----
# 
# This notebook provides a demonstration of how to read data and plot sections to
# understand the temperature/salinity structure of ocean water.
# 
# You will have to install anaconda python to easily setup the required packages.
# If you don't want to install the entire distribution, you can install miniconda
# and install only those packages that you need.
# 
# You will need to install at least xarray, netCDF4 and [gsw](https://teos-10.github.io/GSW-Python/install.html)
# along with the basic packages such as numpy/matplotlib etc., to get started.

# In[4]:


import xarray
import gsw
import numpy as np
import matplotlib.pyplot as plt


# Read in the data and see the coordinates and variables it contains. The description of this
# data is available [here](https://icdc.cen.uni-hamburg.de/en/woce-climatology.html) and [here](ftp://ftp-icdc.cen.uni-hamburg.de/WOCE/climatology/observed_data/BSH35_report_final.pdf) (skim over it!)

# In[6]:


data_list = []
for year in range(2010,2015):
    for month in range(1,13):
        if month<10:
            mon = '0'+str(month)
        else:
            mon = str(month)
        data = xarray.open_dataset('SEA_SURFACE_HEIGHT_mon_mean_{y}-{m}_ECCO_V4r4_latlon_0p50deg.nc'.format(y=year,m=mon))
        data_new = data.loc[dict(longitude=slice(-180, 180), latitude=slice(-90, 90))].SSH
        data_list.append(data_new[0])
data_arr=np.array(data_list);



        


# In[11]:



for i in range(1,6):
    if i==1:
        yearly_summed_data = data_arr[12*(i-1):12*(i),:,:]
    else:
        yearly_summed_data = yearly_summed_data + data_arr[12*(i-1):12*(i),:,:]
        
yearly_avg_data = yearly_summed_data/5

print(np.min(data_arr),np.max(data_arr))
print(np.min(yearly_summed_data),np.max(yearly_summed_data))
print(np.min(yearly_avg_data),np.max(yearly_avg_data))
print(yearly_summed_data.shape)
print(yearly_avg_data.shape)
    
# data_arr.SSH.plot.contourf(levels=np.linspace(-2, 30, 20))


# In[3]:


data


# In[108]:


yav_data = yearly_avg_data[7]

print(np.max((yearly_avg_data[7]==yearly_avg_data[11])*1))
yav_data.shape
yav_data[:,0].shape

print(len(yav_data[0,:]))

arr1 = np.random.rand(360,720)
arr2 = np.random.rand(360,1)
print('Test')
print(arr1.shape, arr2.shape)
print(np.concatenate((arr1, arr2), axis=1).shape)

reshaped_arr = yav_data[:,0].reshape(len(yav_data[:,0]),1)
lon_data_proc = np.concatenate((yav_data, reshaped_arr), axis=1)
print('Error')
print(yav_data.shape,reshaped_arr.shape)
print(lat_data_proc.shape)

lon_diff_data = np.diff(lon_data_proc, axis=1)
print(lon_diff_data.shape)

lat_data_proc = np.concatenate((yav_data, yav_data[0,:].reshape(1,len(yav_data[0,:]))), axis=0)
# print(lat_data_proc.shape)
lat_diff_data = np.diff(lon_data_proc, axis=0)
# print(np.max((lon_diff_data==lat_diff_data)*1))
# print(((lon_data_proc==lat_data_proc)))
# print(lat_diff_data.shape)


# In[62]:


reshaped_arr = yav_data[:,0].reshape(len(yav_data[:,0]),1)
lon_data_proc = np.concatenate((yav_data, yav_data[:,0].reshape(len(yav_data[:,0]),1)), axis=1)

lat_list = np.linspace(-89.75, 89.75, 360)
lon_list = np.linspace(-179.75, 179.75, 720);

omega = 7.2921*(10**-5)

f=2*omega*np.sin(np.deg2rad(lat_list))
g = 9.8
dlat = 0.5
dlon = 0.5

lat_grad = (-g/f.reshape(len(f),1))*(lat_diff_data/dlat)   #.reshape(len(lat_list),1))
lon_grad = (-g/f.reshape(len(f),1))*(lon_diff_data/dlat)   #.reshape(1,len(lon_list)))


# In[49]:


import numpy as np
import matplotlib.pyplot as plt

X, Y = np.meshgrid(xlist, ylist);

Z_lat = lat_grad
Z_lon = lon_grad

fig=plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(1,2,1)
cp = ax1.contourf(X, Y, Z_lon)
ax1.set_title('Filled Contours Plot')
ax1.set_xlabel('x (cm)')
ax1.set_ylabel('y (cm)')

ax2 = fig.add_subplot(1,2,2)
cp = ax2.contourf(X, Y, Z_lon)
ax2.set_title('Filled Contours Plot')
ax2.set_xlabel('x (cm)')
ax2.set_ylabel('y (cm)')

fig.colorbar(cp) # Add a colorbar to a plot

# fig,ax=plt.subplots(1,2,1)
# cp = ax.contourf(X, Y, Z_lon)
# fig.colorbar(cp) # Add a colorbar to a plot
# ax.set_title('Filled Contours Plot')
# ax.set_xlabel('x (cm)')
# ax.set_ylabel('y (cm)')

plt.show()


# In[6]:


x=data.SSH


# In[7]:





# In[67]:


get_ipython().run_line_magic('matplotlib', 'notebook')
data.loc[dict(longitude=slice(-120.25, 120.75), latitude=slice(15.75, 50.75))].SSH.plot


# Plot the temperature at different levels (by selecting the relevant vertical coordinate).
# 
# Here you will see the water temperature at 1000 meters below the surface. Can you interpret the 
# patterns? The textbook (section 13.4) and the assigned book chapter should help!

# In[63]:


get_ipython().run_line_magic('matplotlib', 'notebook')
data.loc[dict(longitude=slice(-180, 180), latitude=slice(-90, 90))].SSH


# The following function is an illustration to calculate quantities of physical interest from
# the available data. See the GSW documentation to find further functions that you might want to
# use.

# In[6]:


def get_density(salinity, temperature, pressure, longitude, latitude):
    
    SA = gsw.SA_from_SP(salinity, pressure, longitude, latitude)
    CT = gsw.CT_from_t(SA, temperature, pressure)
    
    return gsw.density.rho(SA, CT, pressure)


# Plot a vertical section in Temperature-Salinity space. Read about how to interpret this in the textbook and the assigned book chapter.
# 
# The colors are the density which helps us identify different water masses.

# In[7]:


get_ipython().run_line_magic('matplotlib', 'notebook')

vert_section = data.loc[dict(LAT=40, LON=200)]
density = get_density(vert_section.Salinity, vert_section.Temp, vert_section.Pres,
                      vert_section.LON, vert_section.LAT)

lat_grid, lon_grid = np.meshgrid(vert_section.LAT, vert_section.LON)

lat_grid = np.broadcast_to(lat_grid, [44, lat_grid.shape[0], lat_grid.shape[1]])/10

plt.scatter(vert_section.Salinity, vert_section.Temp, s=lat_grid, c=density, cmap='PuOr')


# You can do the same for a region to see what water masses exists in a particular part of the
# ocean. The below plot is for the North Atlantic.

# In[22]:


get_ipython().run_line_magic('matplotlib', 'notebook')
lat_start=-20
lat_end =40

vert_section = data.loc[dict(LAT=slice(-40, 10), LON=slice(310, 330))]
density = get_density(vert_section.Salinity, vert_section.Temp, vert_section.Pres,
                      vert_section.LON, vert_section.LAT)

lat_grid, lon_grid = np.meshgrid(vert_section.LAT, vert_section.LON)

lat_grid = np.broadcast_to(lat_grid, [44, lat_grid.shape[0], lat_grid.shape[1]])/10

plt.scatter(vert_section.Salinity, vert_section.Temp, s=lat_grid, c=density, cmap='PuOr')


# Similarly, you can plot Neutral density surfaces in the ocean basins. It makes sense to take a mean
# across some latitudes. We have ignored the first two hundred meters by slicing `ZAX` to those levels we are
# interested in. A similar slicing along `LON` allows us to choose the Pacific Basin (see lat-lon plot of temperature to find approximate longitudinal extents of ocean basins)

# In[26]:


get_ipython().run_line_magic('matplotlib', 'notebook')

pacific = data.loc[dict(LON=slice(150, 250), ZAX=slice(200, 6000))]

pacific.Gamman.mean(dim=['LON']).plot.contourf(levels=30)
plt.ylim(6000, 0)


# You can do the same for any variable in the dataset. Here we plot potential temperature. Regions without colour usually represent continents or mid-ocean topography.

# In[24]:


get_ipython().run_line_magic('matplotlib', 'notebook')

pacific = data.loc[dict(LON=slice(150, 250), ZAX=slice(100, 6000))]

pacific.Tpoten.mean(dim=['LON']).plot.contourf(levels=np.linspace(0, 30, 20))
plt.ylim(6000, 0)


# # Starting Points
# --------
# 
# Select (at least) two regions (explicitly mention the same) from two different ocean basins and maybe different (north/south) hemispheres.
# 
# * Show how thermo/halocline changes with latitude.
# * Using T/S plots, identify regions/depths/basins with high/low mixing.
# * Plot Neutral density surfaces and infer the circulation in the basins.
# * Plot large-scale variations with depth of temperature/salinity in the world ocean (like above) and interpret the resulting plots.
# * Compare and contrast the characteristics for the two selected regions in terms of ocean-atmosphere interaction (which sets surface properties) and large scale circulation (use Chapter 6 in the textbook along with Aditi's lectures).
