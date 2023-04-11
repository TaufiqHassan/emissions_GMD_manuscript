#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 15:38:32 2023

@author: hass877
"""

## Manuscript Figures

import cartopy
import cartopy.mpl.geoaxes
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import cmaps
import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as crs
import seaborn as sns
import matplotlib.gridspec as gridspec
from manuscript_plot_functions import get_lines, get_plots, rounding, get_stat
from manuscript_plot_functions import get_nearestlatlon, get_mod_alt, getVmap_alt
from manuscript_plot_functions import get_ts, get_vertint, get_var, get_scatter_plot2
from manuscript_plot_functions import get_emisMasked, get_ElevEmisMasked, get_scatter_plot3
plt.rc('font', family='Helvetica')

#####################
## Figure 1
#####################
grid=xr.open_dataset('Data_manuscript/northamericax4v1pg2.g')
fig=plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection=crs.Orthographic(central_latitude=50,central_longitude=-110))
ax.stock_img()
ax.coastlines()
lines=get_lines(grid,ax)
ax.text(-0.03,0.95,'(b) RRM setup',size=20,transform=ax.transAxes)

axins = inset_axes(ax, width="100%", height="100%", loc="upper left", \
                   axes_class = cartopy.mpl.geoaxes.GeoAxes, \
                   axes_kwargs = dict(map_projection = cartopy.crs.PlateCarree()))

ip = InsetPosition(ax,[-1.02, -0.1, 1.2, 1.2])
axins.set_axes_locator(ip)

#####################
data=xr.open_dataset('Data_manuscript/LANDFRAC_ne30pg2.nc')['LANDFRAC'][0]
rr  = np.arange(1,10)
get_plots( data*0,ax=axins,cmap=cmaps.WhiteYellowOrangeRed,levels=rr,\
                 scrip_file='Data_manuscript/ne30pg2_SCRIP.nc',gridLines=True,\
                    lon_range=[-170,-50], lat_range=[15,75], xint=30,
                    unit='',colbar=False).get_map()
## Get site locations
improve = xr.open_dataset('Data_manuscript/aerosol_IMPROVE.nc')
aeronet = xr.open_dataset('Data_manuscript/aod_aeronet_aod_500nm.nc')
## plot locations
i=0
for item,site,name in zip(improve['siteloc'],improve['site'],improve['sitecode']):
    lon,lat=item.values
    if (lat>=17) and (lat<=77) and (lon<=-50) and (lon>=-170):
        i+=1
    axins.scatter(lon,lat,s=25,c='#377eb8',edgecolor='k')
axins.scatter(lon,lat,s=25,c='#377eb8',edgecolor='k',label='IMPROVE')
for item,site,name in zip(aeronet['siteloc'],aeronet['site'],aeronet['sitename']):
    lon,lat=item.values
    if (lat>=17) and (lat<=77) and (lon<=-50) and (lon>=-170):
        i+=1
    axins.scatter(lon,lat,s=20,c='#ff7f00',edgecolor='k',marker='D')
axins.scatter(lon,lat,s=20,c='#ff7f00',edgecolor='k',marker='D',label='AERONET')
axins.text(0.005,1.05,'(a) LR setup with obs sites',size=20,transform=axins.transAxes)
plt.setp(ax.spines.values(),lw=1.5)
axins.legend(loc='lower left',framealpha=1,fontsize=15)
## Save plots
# plt.savefig('fig01.png',format='png',dpi=300,bbox_inches='tight',pad_inches=0.1)
# plt.savefig('fig01.eps',format='eps',dpi=300,bbox_inches='tight',pad_inches=0.1)
# plt.savefig('fig01.pdf',format='pdf',dpi=300,bbox_inches='tight',pad_inches=0.1)

###############################################################
## Figure 2
###############################################################
# From diagram.net >> Full_PFD.drawio


###############################################################
## Figure 3
###############################################################
area_orig=xr.open_dataset('Data_manuscript/gridarea_CEDS.nc')['cell_area']
## factors
inds=[38201, 38084, 37268, 37900, 40145]
inds=[33056, 36078, 37268, 37900, 40145]
lats = [42,40.7,41.8,43.6,45.5]
lons = [-71,-74,-87.6,-79.3,-73.56]
avgod = 6.022e+23
factaa = 1e4*12*1e-3/avgod
factaaa  = 115*10.0/avgod
factbb = 86400.0*365.0*1e-6
plt.rc('font', family='Helvetica')
## CEDS BB BC 1850
lab=['(a)','(b)','(c)','(d)','(e)','(f)','(g)']
ttl=['HR emissions\nRLL 0.5x0.5 deg','LR emissions\nRLL 1.9x2.5 deg','','']
fig=plt.figure(figsize=(16,10),dpi=350)
i=1
for aer in ['bc_a4','so2']:
    for res, grd in zip(['','LR','HR'],['','96x144_northamericax4v1pg2_bl','384x576_northamericax4v1pg2']):
        if i==1:
            bb=xr.open_dataset('Data_manuscript/'+aer+'_HR_emis_2014.nc')[aer]
        elif i<4:
            bb=xr.open_dataset('Data_manuscript/'+aer+'_'+res+'_emis_2014_'+grd+'.nc')[aer]
        elif i==4:
            bb=xr.open_dataset('Data_manuscript/'+aer+'_elev_HR_emis_2014.nc')[aer]
        else:
            bb=xr.open_dataset('Data_manuscript/'+aer+'_elev_'+res+'_emis_2014_'+grd+'.nc')[aer]
        ## plot
        ax=plt.subplot(230+i,projection=crs.PlateCarree())
        rr=[0,0.000274,0.00307,0.0214,0.0793,.198,.392,.682,1.13,5,10,32.9]
        rr = np.array(rr)/(factbb*area_orig.mean().values)
        if aer=='bc_a4':
            bb_factored = bb*factaa
            try:
                val=0
                data = bb_factored.copy()
                lon = xr.where(data.lon > 180,data.lon-360,data.lon)
                lon = lon.assign_coords(lon=lon.values)
                data['lon'] = lon
                lon = lon.sortby(lon)
                data = data.sortby('lon')
                for lt,ln in zip(lats,lons):
                    val+=data.sel(lat=lt,lon=ln,method='nearest').values
                print(val)
            except:
                print(bb_factored[inds].sum().values)
        elif aer=='so2':
            bb_factored = bb*factaa

        if i==1:
            f2=get_plots( bb_factored,ax=ax,cmap=cmaps.amwg256,levels=rr,\
                         scrip_file='',gridLines=True,xint=10,yint=10,\
                            lon_range=[-95,-65], lat_range=[25,55],cbs=5,cbe=-20,cbi=2,res='50m',
                            unit='[Kg m$^{-2}$ s$^{-1}$]',colbar=False).get_map()
            ax.text(0.005,1.05,'Original',size=20,transform=ax.transAxes)
        elif i==4:
            f2=get_plots( bb_factored,ax=ax,cmap=cmaps.amwg256,levels=rr,\
                         scrip_file='',gridLines=True,xint=10,yint=10,\
                            lon_range=[-130,-100], lat_range=[25,55],cbs=5,cbe=-20,cbi=2,res='50m',
                            unit='[Kg m$^{-2}$ s$^{-1}$]',colbar=False).get_map()
            ax.text(0.005,1.05,'Original',size=20,transform=ax.transAxes)
        elif (i==2) or (i==3):
            get_plots( bb_factored,ax=ax,cmap=cmaps.amwg256,levels=rr,\
                         scrip_file='Data_manuscript/northamericax4v1pg2_scrip.nc',gridLines=True,xint=10,yint=10,\
                            lon_range=[-95,-65], lat_range=[25,55],cbs=5,cbe=-20,cbi=2,res='50m',
                            unit='[Kg m$^{-2}$ s$^{-1}$]',colbar=False).get_map()
        else:
            get_plots( bb_factored,ax=ax,cmap=cmaps.amwg256,levels=rr,\
                         scrip_file='Data_manuscript/northamericax4v1pg2_scrip.nc',gridLines=True,xint=10,yint=10,\
                            lon_range=[-130,-100], lat_range=[25,55],cbs=5,cbe=-20,cbi=2,res='50m',
                            unit='[Kg m$^{-2}$ s$^{-1}$]',colbar=False).get_map()
        if i==2:
            ax.scatter(-71,42,s=30,c='red',edgecolor='k',marker='o',transform=crs.PlateCarree(),zorder=4)
            ax.text(-71+.3,42,'1',size=10,transform=crs.PlateCarree(),va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'},zorder=3)
            ax.scatter(-74,40.7,s=30,c='red',edgecolor='k',marker='o',transform=crs.PlateCarree(),zorder=4)
            ax.text(-74+.3,40.7,'2',size=10,transform=crs.PlateCarree(),va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'},zorder=3)
            ax.scatter(-87.6,41.8,s=30,c='red',edgecolor='k',marker='o',transform=crs.PlateCarree(),zorder=4)
            ax.text(-87.6+.3,41.8,'3',size=10,transform=crs.PlateCarree(),va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'},zorder=3)
            ax.scatter(-79.3,43.6,s=30,c='red',edgecolor='k',marker='o',transform=crs.PlateCarree(),zorder=4)
            ax.text(-79.3+.3,43.6,'4',size=10,transform=crs.PlateCarree(),va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'},zorder=3)
            ax.scatter(-73.56,45.5,s=30,c='red',edgecolor='k',marker='o',transform=crs.PlateCarree(),zorder=4)
            ax.text(-73.56+.3,45.5,'5',size=10,transform=crs.PlateCarree(),va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'},zorder=3)
        if i==5:
            ax.scatter(-118.2,34,s=30,c='red',edgecolor='k',marker='o',transform=crs.PlateCarree(),zorder=4)
            ax.text(-118.2+.4,34,'6',size=10,transform=crs.PlateCarree(),va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'},zorder=3)
            ax.scatter(-122.4,37.7,s=30,c='red',edgecolor='k',marker='o',transform=crs.PlateCarree(),zorder=4)
            ax.text(-122.4+.4,37.7,'7',size=10,transform=crs.PlateCarree(),va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'},zorder=3)
        if (i==2) or (i==5):
            ax.text(0.005,1.05,'Default treatment',size=20,transform=ax.transAxes)
        elif (i==3) or (i==6):
            ax.text(0.005,1.05,'Improved treatment',size=20,transform=ax.transAxes)
        ax.text(0.05,0.95,lab[i-1],size=20,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
        plt.setp(ax.spines.values(),lw=2)
        if i==1:
            plt.ylabel('Surface BC',fontsize=20)
        if i==4:
            plt.ylabel('Column integrated SO$_2$',fontsize=20)
        plt.setp(ax.spines.values(),lw=2)
        i+=1
## colorbar setup
fig.subplots_adjust(top=0.95)
s1 = pd.DataFrame(rr)
s2 = s1.applymap(lambda x: rounding(x))[0].tolist()
cbar_ticks=list(map(str,s2))
cbar_ticks = [i.rstrip('.').strip('e+') for i in cbar_ticks]
cbar_ticks[0]=''
cbar_ticks[-1]=''
cbar_ax = fig.add_axes([0.125,0.06,0.775,0.03])
cbar = fig.colorbar(f2,cax=cbar_ax,pad=0.12,orientation='horizontal',extend='neither',ticks=rr,drawedges=True)
cbar.ax.set_xticklabels(cbar_ticks)
cbar.set_label(label='[Kg m$^{-2}$ s$^{-1}$]',size=15)
cbar.outline.set_linewidth(1.5)
cbar.dividers.set_linewidth(1.5)

# plt.savefig('RRM_emissions_comparison_Default_vs_SE_Manuscript_7Points.png',dpi=300,format='png',bbox_inches='tight',pad_inches=0.1)
# plt.savefig('RRM_emissions_comparison_Default_vs_SE_Manuscript_7Points.pdf',dpi=350,format='pdf',bbox_inches='tight',pad_inches=0.1)

###############################################################
## Figure 4
###############################################################
data = xr.open_dataset('Data_manuscript/F20TR_v2_ndg_ERA5_SEdata_NA_RRM_CDT.eam.h1.2016-01-01-00000.nc')
lon = data['lon'].values
lon[lon > 180.] -= 360.
newlon = xr.DataArray(lon,coords={'ncol':data.ncol.values})
lat = data['lat']        
        
i=1
lab=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
fig=plt.figure(figsize=(14,16.85))
for aer, tt, tt2 in zip(['bc','pom'],['RRM-PD (Default treatment)',''],['Relative difference','']):
    df=xr.open_dataset('Data_manuscript/'+aer+'_F20TR_v2_ndg_ERA5_Defdata_NA_RRM_CDT_hplot_0.nc')[aer]
    df = df.assign_coords(lon=newlon)
    df = df.assign_coords(lat=lat)
    se=xr.open_dataset('Data_manuscript/'+aer+'_F20TR_v2_ndg_ERA5_SEdata_NA_RRM_CDT_hplot_0.nc')[aer]
    se = se.assign_coords(lon=newlon)
    se = se.assign_coords(lat=lat)
    astats = (get_stat(df,se,weight=1))
    if i==1:
        rr=[0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 1.0]
    else:
        rr=[0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0]
    print(rr)
    diff = se-df
    rel = (diff/abs(df))*100
    rr_rel=[-100.0,-50.0,-20.0,-10.0,-5.0,-2.0,2.0,5.0,10.0,20.0,50.0,100.0]
    ax=plt.subplot(420+i,projection=crs.PlateCarree())
    get_plots( df,ax=ax,cmap=cmaps.amwg256,levels=rr,res='10m',bdist=0.03,th=0.01,\
                 scrip_file='Data_manuscript/northamericax4v1pg2_scrip.nc',gridLines=False,\
                    lon_range=[-170,-50], lat_range=[15,75],xint=30,yint=15,cbs=5,cbe=-20,cbi=2,
                    unit='[$\u03BCg\ m^{-3}$]',colbar=True).get_map()
    ax.text(0.12,1.15,tt,size=20,transform=ax.transAxes,weight='bold')
    ax.text(0.05,0.95,lab[i-1],size=20,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
    ax.text(0.8,0.95,'Mean: '+str(rounding(astats[0])),size=12,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
    if i==1:
        ax.text(-0.2,0.45,' BC',size=20,transform=ax.transAxes,rotation=90)
    else:
        ax.text(-0.2,0.45,'POM',size=20,transform=ax.transAxes,rotation=90)

    ax=plt.subplot(420+i+1,projection=crs.PlateCarree())
    get_plots( rel,ax=ax,cmap=cmaps.BlueWhiteOrangeRed,levels=rr_rel,res='10m',bdist=0.03,th=0.01,\
                 scrip_file='Data_manuscript/northamericax4v1pg2_scrip.nc',gridLines=False,\
                    lon_range=[-170,-50], lat_range=[15,75],xint=30,yint=15,
                    unit='[%]',colbar=True).get_map()
    ax.text(0.21,1.15,tt2,size=20,transform=ax.transAxes,weight='bold')
    ax.text(0.05,0.95,lab[i],size=20,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
    ax.text(0.74,0.95,'RMSE: '+str(rounding(astats[1])),size=12,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
    ax.text(0.74,0.875,'N_RMSE: '+str(round(astats[2]*100))+'%',size=12,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})

    i=i+2
for aer, tt, tt2 in zip(['so4'],['RRM-PD (Default treatment)'],['Rel diff']):
    df=xr.open_dataset('Data_manuscript/'+aer+'_F20TR_v2_ndg_ERA5_Defdata_NA_RRM_CDT_hplot_0.nc')[aer]
    df = df.assign_coords(lon=newlon)
    df = df.assign_coords(lat=lat)
    se=xr.open_dataset('Data_manuscript/'+aer+'_F20TR_v2_ndg_ERA5_SEdata_NA_RRM_CDT_hplot_0.nc')[aer]
    se = se.assign_coords(lon=newlon)
    se = se.assign_coords(lat=lat)
    astats = (get_stat(df,se,weight=1))
    if i==5:
        rr=[0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3, 4, 5.0, 10.0, 20.0]
    else:
        rr=[0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    print(rr)
    diff = se-df
    rel = (diff/abs(df))*100
    rr_rel=[-100.0,-50.0,-20.0,-10.0,-5.0,-2.0,2.0,5.0,10.0,20.0,50.0,100.0]
    ax=plt.subplot(420+i,projection=crs.PlateCarree())
    get_plots( df,ax=ax,cmap=cmaps.amwg256,levels=rr,res='10m',\
                 scrip_file='Data_manuscript/northamericax4v1pg2_scrip.nc',gridLines=False,bdist=0.03,th=0.01,\
                    lon_range=[-170,-50], lat_range=[15,75],xint=30,yint=15,cbs=5,cbe=-20,cbi=2,
                    unit='[ug m$^{-3}$]',colbar=True).get_map()
    ax.text(0.05,0.95,lab[i-1],size=20,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
    ax.text(0.8,0.95,'Mean: '+str(rounding(astats[0])),size=12,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
    if i==5:
        ax.text(-0.2,0.4,'Sulfate',size=20,transform=ax.transAxes,rotation=90)
    else:
        ax.text(-0.2,0.45,'SO$_2$',size=20,transform=ax.transAxes,rotation=90)

    ax=plt.subplot(420+i+1,projection=crs.PlateCarree())
    get_plots( rel,ax=ax,cmap=cmaps.BlueWhiteOrangeRed,levels=rr_rel,res='10m',bdist=0.03,th=0.01,\
                 scrip_file='Data_manuscript/northamericax4v1pg2_scrip.nc',gridLines=False,\
                    lon_range=[-170,-50], lat_range=[15,75],xint=30,yint=15,
                    unit='[%]',colbar=True).get_map()
    ax.text(0.05,0.95,lab[i],size=20,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
    ax.text(0.74,0.95,'RMSE: '+str(rounding(astats[1])),size=12,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
    ax.text(0.74,0.875,'N_RMSE: '+str(round(astats[2]*100))+'%',size=12,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
    i=i+2
# plt.savefig('fig04.pdf',dpi=300,format='pdf',bbox_inches='tight',pad_inches=0.1)
# plt.savefig('fig04.png',dpi=300,format='png',bbox_inches='tight',pad_inches=0.1)
###############################################################
## Figure 5
###############################################################
data = pd.read_excel('Data_manuscript/barplot_manuscript.xlsx',index_col='AA')
bc = data[:4][::-1][['Major cities','Within 75/25 percentile','<25 percentile','>75 percentile']]

pom = data[4:8]
pom = data[4:8][::-1][['Major cities','Within 75/25 percentile','<25 percentile','>75 percentile']]
so4 = data[8:12]
so4 = data[8:12][::-1][['Major cities','Within 75/25 percentile','<25 percentile','>75 percentile']]

fig=plt.figure(figsize=(18,8))
ax1=plt.subplot(131)
hh=0.2
ax1.barh(np.arange(0, len(bc)), bc['Major cities'], color='lightgray',height=hh,edgecolor='k', hatch='//',zorder=4)
ax1.barh(np.arange(hh, len(bc)+hh), bc['Within 75/25 percentile'], color='dimgray',edgecolor='k', height=hh,hatch='--',zorder=4)
ax1.barh(np.arange(2*hh, len(bc)+2*hh), bc['<25 percentile'], color='lightgray', height=hh,edgecolor='k', hatch='..',zorder=4)
ax1.barh(np.arange(3*hh, len(bc)+3*hh), bc['>75 percentile'], color='lightgray', height=hh,edgecolor='k',zorder=4)
#ax1.get_legend().remove()
plt.setp(ax1.spines.values(),lw=1.5)
ax1.grid(linestyle='--',color='#EBE7E0',zorder=3)
plt.tick_params(labelsize=20)
ax1.axvline(x=0,c='k',lw=0.5,linestyle='--')
plt.ylabel('')
ax1.text(0.05,0.95,'(a)',size=20,transform=ax1.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
ax1.text(0.01,1.05,'BC',size=20,transform=ax1.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
#plt.xlim([-100,100])
plt.yticks([0.25,1.25,2.25,3.25],['Sinks','Sources','Sfc conc.','Burden'])


ax1=plt.subplot(132)
#pom.plot.barh(ax=ax1,zorder=4,color=['#4daf4a','#f781bf','#377eb8','#ff7f00'],linewidth=100,capstyle='butt')
ax1.barh(np.arange(0, len(pom)), pom['Major cities'], color='lightgray',height=hh,edgecolor='k', hatch='//',zorder=4)
ax1.barh(np.arange(hh, len(pom)+hh), pom['Within 75/25 percentile'], color='dimgray',edgecolor='k', height=hh,hatch='--',zorder=4)
ax1.barh(np.arange(2*hh, len(pom)+2*hh), pom['<25 percentile'], color='lightgray', height=hh,edgecolor='k', hatch='..',zorder=4)
ax1.barh(np.arange(3*hh, len(pom)+3*hh), pom['>75 percentile'], color='lightgray', height=hh,edgecolor='k',zorder=4)
#ax1.get_legend().remove()
plt.ylabel('')
plt.yticks([0.25,1.25,2.25,3.25],['Sinks','Sources','Sfc conc.','Burden'])
ax1.tick_params(axis='y',which='both',left=False)
ax1.grid(linestyle='--',color='#EBE7E0',zorder=3)
ax1.set_yticklabels(['','','',''])
plt.setp(ax1.spines.values(),lw=1.5)
plt.tick_params(labelsize=20)
ax1.axvline(x=0,c='k',lw=0.5,linestyle='--')
plt.xlabel('Relative difference (%)',fontsize=20)
ax1.text(0.05,0.95,'(b)',size=20,transform=ax1.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
ax1.text(0.01,1.05,'POM',size=20,transform=ax1.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
#plt.xlim([-100,100])

ax1=plt.subplot(133)
#so4.plot.barh(ax=ax1,zorder=4,color=['#4daf4a','#f781bf','#377eb8','#ff7f00'],linewidth=100,capstyle='butt')
ax1.barh(np.arange(0, len(so4)), so4['Major cities'], color='lightgray',height=hh,edgecolor='k', hatch='//',zorder=4,label='Major cities')
ax1.barh(np.arange(hh, len(so4)+hh), so4['Within 75/25 percentile'], color='dimgray',edgecolor='k', height=hh,hatch='--',zorder=4,label='Within 75/25 percentile')
ax1.barh(np.arange(2*hh, len(so4)+2*hh), so4['<25 percentile'], color='lightgray', height=hh,edgecolor='k', hatch='..',zorder=4,label='<25 percentile')
ax1.barh(np.arange(3*hh, len(so4)+3*hh), so4['>75 percentile'], color='lightgray', height=hh,edgecolor='k',zorder=4,label='>75 percentile')
plt.ylabel('')
plt.yticks([0.25,1.25,2.25,3.25],['Sinks','Sources','Sfc conc.','Burden'])
ax1.tick_params(axis='y',which='both',left=False)
ax1.grid(linestyle='--',color='#EBE7E0',zorder=3)
ax1.set_yticklabels(['','','',''])
plt.setp(ax1.spines.values(),lw=1.5)
ax1.axvline(x=0,c='k',lw=0.5,linestyle='--')
plt.tick_params(labelsize=20)
ax1.text(0.05,0.95,'(c)',size=20,transform=ax1.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
ax1.text(0.01,1.05,'Sulfate',size=20,transform=ax1.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
plt.legend()
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[::-1], labels[::-1],loc='upper right',fontsize=15,frameon=False)
#plt.xlim([-100,100])
plt.tight_layout()
# plt.savefig('fig05.png',dpi=300,format='png',bbox_inches='tight',pad_inches=0.1)
# plt.savefig('fig05.pdf',dpi=300,format='pdf',bbox_inches='tight',pad_inches=0.1)
# plt.savefig('fig05.eps',dpi=300,format='eps',bbox_inches='tight',pad_inches=0.1)
###############################################################
## Figure 6
###############################################################
data = xr.open_dataset('Data_manuscript/F20TR_v2_ndg_ERA5_SEdata_NA_RRM_CDT.eam.h1.2016-01-01-00000.nc')
lon = data['lon'].values
lon[lon > 180.] -= 360.
grav = 9.806
ha = data['hyai']
hb = data['hybi']
p0 = data['P0']
ps = data['PS'][0]
area = data['area']
fact=1e9
factaa = 1.01325e5 / 8.31446261815324 / 273.15 * 28.9647 / 1.e9   # kg-air/cm3-air
factbb = factaa * 1.e15  # ug-air/m3-air
lat1,lon1 = 40.5,-74.5 # SO4 AE HE
alt = np.array([50.,80.,100.,200.,300.,400.,500.,600,700,800,900,1000.,2000.,3000.,4000.,5000.,10000.])
rr=[2e-14,5e-14,1e-13,2e-13,5e-13,1e-12,2e-12,5e-12,1e-11,2e-11,5e-11,1e-10,1.5e-10,2e-10,3e-10,4e-10,5e-10,1e-9]
rr_rel=[-1e3,-100.,-80,-65,-50.,-20.,-10.,-5.,-2.,2.,5.,10.,20.,50.,65,80,100.,1e3]
ind=37900
dataSE = xr.open_dataset('Data_manuscript/ERA5_SEdata_NA_RRM_bc_total_daily_july.nc')['bc']
datadef = xr.open_dataset('Data_manuscript/ERA5_Defdata_NA_RRM_bc_total_daily_july.nc')['bc']

valsSE = get_mod_alt(dataSE,ind,alt)
valsdef = get_mod_alt(datadef,ind,alt)
valdiff = valsSE - valsdef
valrel = (valdiff/abs(valsdef))*100
burdendef = get_vertint(datadef,ha,p0,hb,ps,grav,fact)
burdenSE = get_vertint(dataSE,ha,p0,hb,ps,grav,fact)
burdendiff = burdenSE - burdendef
burdenrel = (burdendiff/abs(burdendef))*100

fig = plt.figure(figsize=(14, 16))
gs = gridspec.GridSpec(14, 9,wspace=2)

ax1 = fig.add_subplot(gs[:5, :4])
getVmap_alt(valsdef[:,:],rr,ax1,'[kg/kg]',cmaps.amwg256[:-20],alt,fig)
ax1.text(0.02,0.95,'(a)',size=20,transform=ax1.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
ax1.text(0.005,1.05,'RRM-PD (Default treatment)',size=20,transform=ax1.transAxes)
ax1.text(-0.4,-0.1,'Highly polluted location',rotation=90,size=20,transform=ax1.transAxes)
ax2 = fig.add_subplot(gs[5:7, :4],sharex=ax1)
ax1.set_ylabel('Concentration\nAltitude [m]',fontsize=15)
get_ts(burdendef,ax2,'[$\u03BCg\ m^{-2}$]',ind)
ax2.text(0.02,0.95,'(b)',size=20,transform=ax2.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
ax2.set_ylabel('Burden\n[$\u03BCg\ m^{-2}$]',fontsize=15)

ax1 = fig.add_subplot(gs[:5, 5:9])
getVmap_alt(valrel[:,:],rr_rel,ax1,'[%]',cmaps.BlueWhiteOrangeRed[10:-10],alt,fig)
ax1.text(0.02,0.95,'(c)',size=20,transform=ax1.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
ax1.text(0.005,1.05,'Relative difference',size=20,transform=ax1.transAxes)
#ax2 = fig.add_subplot(gs[5:, 5:],sharex=ax1)
ax2 = fig.add_subplot(gs[5:7, 5:9],sharex=ax1)
get_ts(burdenrel,ax2,'[%]',ind)
ax2.set_ylabel('[%]',fontsize=15)
ax2.text(0.02,0.95,'(d)',size=20,transform=ax2.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})

factaa = 1.01325e5 / 8.31446261815324 / 273.15 * 28.9647 / 1.e9   # kg-air/cm3-air
factbb = factaa * 1.e15  # ug-air/m3-air
lats = [34,37.77,41.8,43.6,45.5]
lons = [-118.2,-122.4,-87.6,-79.3,-73.56]
lat1,lon1 = 45,-80 # SO4 AE HE
alt = np.array([50.,80.,100.,200.,300.,400.,500.,600,700,800,900,1000.,2000.,3000.,4000.,5000.,10000.])
rr=[2e-14,5e-14,1e-13,2e-13,5e-13,1e-12,2e-12,5e-12,1e-11,2e-11,5e-11,1e-10,1.5e-10,2e-10,3e-10,4e-10,5e-10,1e-9]
rr_rel=[-1e3,-100.,-80,-65,-50.,-20.,-10.,-5.,-2.,2.,5.,10.,20.,50.,65,80,100.,1e3]
lat1,lat2,lon1,lon2,ind = get_nearestlatlon(lon1,lat1,lon,lat)
valsSE = get_mod_alt(dataSE,ind,alt)
valsdef = get_mod_alt(datadef,ind,alt)
valdiff = valsSE - valsdef
valrel = (valdiff/abs(valsdef))*100
burdendef = get_vertint(datadef,ha,p0,hb,ps,grav,fact)
burdenSE = get_vertint(dataSE,ha,p0,hb,ps,grav,fact)
burdendiff = burdenSE - burdendef
burdenrel = (burdendiff/abs(burdendef))*100
#ax1 = fig.add_subplot(gs[:5, :4])
ax1 = fig.add_subplot(gs[7:7+5, :4])
getVmap_alt(valsdef[:,:],rr,ax1,'[kg/kg]',cmaps.amwg256[:-20],alt,fig)
ax1.set_ylabel('Concentration\nAltitude [m]',fontsize=15)
ax1.text(0.02,0.95,'(e)',size=20,transform=ax1.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
ax1.text(-0.4,-0.1,'Nearby less polluted location',rotation=90,size=20,transform=ax1.transAxes)
ax2 = fig.add_subplot(gs[7+5:, :4],sharex=ax1)
get_ts(burdendef,ax2,'[$\u03BCg\ m^{-2}$]',ind)
ax2.set_ylabel('Burden\n[$\u03BCg\ m^{-2}$]',fontsize=15)
ax2.text(0.02,0.95,'(f)',size=20,transform=ax2.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})

ax1 = fig.add_subplot(gs[7:7+5, 5:9])
getVmap_alt(valrel[:,:],rr_rel,ax1,'[%]',cmaps.BlueWhiteOrangeRed[10:-10],alt,fig)
ax1.text(0.02,0.95,'(g)',size=20,transform=ax1.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
#ax1.text(0.005,1.05,'Relative diff',size=20,transform=ax1.transAxes,weight='bold')
ax2 = fig.add_subplot(gs[7+5:, 5:9],sharex=ax1)
get_ts(burdenrel,ax2,'[%]',ind)
ax2.set_ylabel('[%]',fontsize=15)
ax2.text(0.02,0.95,'(h)',size=20,transform=ax2.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})

# plt.savefig('fig06.png',dpi=300,format='png',bbox_inches='tight',pad_inches=0.1)
# plt.savefig('fig06.pdf',dpi=300,format='pdf',bbox_inches='tight',pad_inches=0.1)
# plt.savefig('fig06.eps',dpi=300,format='eps',bbox_inches='tight',pad_inches=0.1)

###############################################################
## Figure 7
###############################################################
data = xr.open_dataset('Data_manuscript/F20TR_v2_ndg_ERA5_SEdata_NA_RRM_CDT.eam.h1.2016-01-01-00000.nc')
lon = data['lon'].values
lon[lon > 180.] -= 360.
newlon = xr.DataArray(lon,coords={'ncol':data.ncol.values})
lat = data['lat']        
        
i=1
lab=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
#fig=plt.figure(figsize=(12,14))
fig=plt.figure(figsize=(14,16.85))
for aer, tt, tt2 in zip(['EXTINCT','ABSORB'],['RRM-PD (Default treatment)',''],['Relative difference','']):
    df=xr.open_dataset('Data_manuscript/'+aer+'_F20TR_v2_ndg_ERA5_Defdata_hplot_0.nc')[aer][-1]*1e3
    df = df.assign_coords(lon=newlon)
    df = df.assign_coords(lat=lat)
    se=xr.open_dataset('Data_manuscript/'+aer+'_F20TR_v2_ndg_ERA5_SEdata_hplot_0.nc')[aer][-1]*1e3
    se = se.assign_coords(lon=newlon)
    se = se.assign_coords(lat=lat)
    astats = (get_stat(df,se,weight=1))
    if i==1:
        rr=[1e-5, 1e-2, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2]
    else:
        rr=[1e-6, 1e-5, 1e-4, 1e-3, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.01, 0.015]
    print(rr)
    diff = se-df
    rel = (diff/abs(df))*100
    rr_rel=[-100.0,-50.0,-20.0,-10.0,-5.0,-2.0,2.0,5.0,10.0,20.0,50.0,100.0]
    ax=plt.subplot(420+i,projection=crs.PlateCarree())
    get_plots( df,ax=ax,cmap=cmaps.amwg256,levels=rr,res='10m',\
                 scrip_file='Data_manuscript/northamericax4v1pg2_scrip.nc',gridLines=False,bdist=0.03,th=0.01,\
                    lon_range=[-170,-50], lat_range=[15,75],xint=30,yint=15,cbs=5,cbe=-20,cbi=2,
                    unit='',colbar=True).get_map()
    ax.text(0.12,1.15,tt,size=20,transform=ax.transAxes,weight='bold')
    ax.text(0.05,0.95,lab[i-1],size=20,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
    ax.text(0.78,0.95,'Mean: '+str(rounding(astats[0])),size=12,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
    if i==1:
        ax.text(-0.23,0.28,'  Extinction\nnear surface',size=20,transform=ax.transAxes,rotation=90)
    else:
        ax.text(-0.23,0.28,'  Absorption\nnear surface',size=20,transform=ax.transAxes,rotation=90)

    ax=plt.subplot(420+i+1,projection=crs.PlateCarree())
    get_plots( rel,ax=ax,cmap=cmaps.BlueWhiteOrangeRed,levels=rr_rel,res='10m',\
                 scrip_file='northamericax4v1pg2_scrip.nc',gridLines=False,bdist=0.03,th=0.01,\
                    lon_range=[-170,-50], lat_range=[15,75],xint=30,yint=15,
                    unit='',colbar=True).get_map()
    ax.text(0.21,1.15,tt2,size=20,transform=ax.transAxes,weight='bold')
    ax.text(0.05,0.95,lab[i],size=20,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
    ax.text(0.74,0.95,'RMSE: '+str(rounding(astats[1])),size=12,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
    ax.text(0.74,0.875,'N_RMSE: '+str(round(astats[2]*100))+'%',size=12,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
    i=i+2
for aer, tt, tt2 in zip(['AODVIS','AODABS'],['RRM-PD (Default treatment)',''],['Relative difference','']):
    df=xr.open_dataset('Data_manuscript/'+aer+'_F20TR_v2_ndg_ERA5_Defdata_NA_RRM_CDT_hplot_0.nc')[aer]
    df = df.assign_coords(lon=newlon)
    df = df.assign_coords(lat=lat)
    se=xr.open_dataset('Data_manuscript/'+aer+'_F20TR_v2_ndg_ERA5_SEdata_NA_RRM_CDT_hplot_0.nc')[aer]
    se = se.assign_coords(lon=newlon)
    se = se.assign_coords(lat=lat)
    astats = (get_stat(df,se,weight=1))
    if i==5:
        rr=[0.01, 0.02, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
    else:
        rr=[0.001 , 0.002 , 0.0025, 0.005 , 0.0075, 0.01  , 0.0125, 0.015 ,0.0175, 0.02  , 0.0225, 0.025 , 0.0275, 0.03]
    print(rr)
    diff = se-df
    rel = (diff/abs(df))*100
    rr_rel=[-100.0,-50.0,-20.0,-10.0,-5.0,-2.0,2.0,5.0,10.0,20.0,50.0,100.0]
    ax=plt.subplot(420+i,projection=crs.PlateCarree())
    get_plots( df,ax=ax,cmap=cmaps.amwg256,levels=rr,res='10m',\
                 scrip_file='Data_manuscript/northamericax4v1pg2_scrip.nc',gridLines=False,bdist=0.03,th=0.01,\
                    lon_range=[-170,-50], lat_range=[15,75],xint=30,yint=15,cbs=5,cbe=-20,cbi=2,
                    unit='',colbar=True).get_map()
    #ax.text(0.12,1.15,tt,size=20,transform=ax.transAxes,weight='bold')
    ax.text(0.05,0.95,lab[i-1],size=20,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
    ax.text(0.78,0.95,'Mean: '+str(rounding(astats[0])),size=12,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
    if i==5:
        ax.text(-0.2,0.4,'AOD',size=20,transform=ax.transAxes,rotation=90)
    else:
        ax.text(-0.23,0.3,'Absorption\n     AOD',size=20,transform=ax.transAxes,rotation=90)

    ax=plt.subplot(420+i+1,projection=crs.PlateCarree())
    get_plots( rel,ax=ax,cmap=cmaps.BlueWhiteOrangeRed,levels=rr_rel,res='10m',\
                 scrip_file='Data_manuscript/northamericax4v1pg2_scrip.nc',gridLines=False,bdist=0.03,th=0.01,\
                    lon_range=[-170,-50], lat_range=[15,75],xint=30,yint=15,
                    unit='',colbar=True).get_map()
    #ax.text(0.21,1.15,tt2,size=20,transform=ax.transAxes,weight='bold')
    ax.text(0.05,0.95,lab[i],size=20,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
    ax.text(0.74,0.95,'RMSE: '+str(rounding(astats[1])),size=12,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
    ax.text(0.74,0.875,'N_RMSE: '+str(round(astats[2]*100))+'%',size=12,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})

    i=i+2

# plt.savefig('fig07.pdf',dpi=300,format='pdf',bbox_inches='tight',pad_inches=0.1)
# plt.savefig('fig07.png',dpi=300,format='png',bbox_inches='tight',pad_inches=0.1)

###############################################################
## Figure 8
###############################################################
alt = np.array([50.,80.,100.,200.,300.,400.,500.,600,700,800,900,1000.,2000.,3000.,4000.,5000.,10000.])
rr=[1e-5, 1e-2, 0.02, 0.03, 0.04, 0.05, .06,.07,.08,.09,0.1, 0.15, 0.2]
rr_rel=[-500.,-100.,-80,-65,-50.,-20.,-10.,-5.,-2.,2.,5.,10.,20.,50.,65,80,100.,500.]
datase = xr.open_dataset('Data_manuscript/ERA5_SEdata_NA_RRM_optics_daily_july.nc')
datadef = xr.open_dataset('Data_manuscript/ERA5_Defdata_NA_RRM_optics_daily_july.nc')
vv1,inds=get_var(datadef,'EXTINCT',reg='TCAP')
vv2,inds=get_var(datase,'EXTINCT',reg='TCAP')
valsSE = get_mod_alt(vv2,ind,alt)
valsdef = get_mod_alt(vv1,ind,alt)
valdiff = valsSE - valsdef
valrel = (valdiff/abs(valsdef))*100
vv1,inds=get_var(datadef,'AODVIS',reg='TCAP')
vv2,inds=get_var(datase,'AODVIS',reg='TCAP')
diff = vv2-vv1
rel = (diff/abs(vv1))*100
######################################
fig = plt.figure(figsize=(14, 16))
gs = gridspec.GridSpec(14, 9,wspace=2)

ax1 = fig.add_subplot(gs[:5, :4])
getVmap_alt(valsdef[:,:]*1e3,rr,ax1,'[km$^{-1}$]',cmaps.amwg256[:-20],alt,fig)
ax1.text(0.02,0.95,'(a)',size=20,transform=ax1.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
ax1.text(0.005,1.05,'RRM-PD (Default treatment)',size=20,transform=ax1.transAxes)
ax2 = fig.add_subplot(gs[5:7, :4],sharex=ax1)
ax1.set_ylabel('Aerosol Extinction\nAltitude [m]',fontsize=15)
get_ts(vv1,ax2,'[1]',ind)
ax2.text(0.02,0.95,'(b)',size=20,transform=ax2.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
ax2.set_ylabel('AOD\n[1]',fontsize=15)

ax1 = fig.add_subplot(gs[:5, 5:9])
getVmap_alt(valrel[:,:],rr_rel,ax1,'[%]',cmaps.BlueWhiteOrangeRed[10:-10],alt,fig)
ax1.text(0.02,0.95,'(c)',size=20,transform=ax1.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
ax1.text(0.005,1.05,'Relative difference',size=20,transform=ax1.transAxes)
ax2 = fig.add_subplot(gs[5:7, 5:9],sharex=ax1)
get_ts(rel,ax2,'[$\u03BCg\ m^{-2}$]',ind)
ax2.axhline(linewidth=0.5,linestyle='--', color='darkgray',zorder=2)
ax2.set_ylabel('[%]',fontsize=15)
ax2.text(0.02,0.95,'(d)',size=20,transform=ax2.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
#########
alt = np.array([50.,80.,100.,200.,300.,400.,500.,600,700,800,900,1000.,2000.,3000.,4000.,5000.,10000.])
rr=[1e-6, 1e-5, 1e-4, 1e-3, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.006,.008, 0.01, 0.015]
rr_rel=[-100.,-80,-65,-50.,-20.,-10.,-5.,-2.,2.,5.,10.,20.,50.,65,80,100.]
vv1,inds=get_var(datadef,'ABSORB',reg='TCAP')
vv2,inds=get_var(datase,'ABSORB',reg='TCAP')
valsSE = get_mod_alt(vv2,ind,alt)
valsdef = get_mod_alt(vv1,ind,alt)
valdiff = valsSE - valsdef
valrel = (valdiff/abs(valsdef))*100
valsdef = valsdef.resample(time="1D").mean(dim='time')
valrel = valrel.resample(time="1D").mean(dim='time')
###
vv1,inds=get_var(datadef,'AODABS',reg='TCAP')
vv2,inds=get_var(datase,'AODABS',reg='TCAP')
vv1 = vv1.resample(time="1D").mean(dim='time')
vv2 = vv2.resample(time="1D").mean(dim='time')
diff = vv2-vv1
rel = (diff/abs(vv1))*100
######################################
ax1 = fig.add_subplot(gs[7:7+5, :4])
getVmap_alt(valsdef[:,:]*1e3,rr,ax1,'[km$^{-1}$]',cmaps.amwg256[:-20],alt,fig)
ax1.set_ylabel('Aerosol Absorption\nAltitude [m]',fontsize=15)
ax1.text(0.02,0.95,'(e)',size=20,transform=ax1.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
ax2 = fig.add_subplot(gs[7+5:, :4],sharex=ax1)
#get_ts(valsdef*1e9,ax2,'[10$^{-9}$ kg/kg]')
get_ts(vv1,ax2,'[1]',ind)
ax2.set_ylabel('AAOD\n[1]',fontsize=15)
ax2.text(0.02,0.95,'(f)',size=20,transform=ax2.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
ax1 = fig.add_subplot(gs[7:7+5, 5:9])
getVmap_alt(valrel[:,:],rr_rel,ax1,'[%]',cmaps.BlueWhiteOrangeRed[10:-10],alt,fig)
ax1.text(0.02,0.95,'(g)',size=20,transform=ax1.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
#ax1.text(0.005,1.05,'Relative diff',size=20,transform=ax1.transAxes,weight='bold')
ax2 = fig.add_subplot(gs[7+5:, 5:9],sharex=ax1)
get_ts(rel,ax2,'[$\u03BCg\ m^{-2}$]',ind)
ax2.axhline(linewidth=0.5,linestyle='--', color='darkgray',zorder=2)
ax2.set_ylabel('[%]',fontsize=15)
ax2.text(0.02,0.95,'(h)',size=20,transform=ax2.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})

# plt.savefig('fig08.png',format='png',dpi=300,bbox_inches='tight',pad_inches=0.1)
# plt.savefig('fig08.pdf',format='pdf',dpi=300,bbox_inches='tight',pad_inches=0.1)
###############################################################
## Figure 9
###############################################################
## Stacked plot
r = [0,1,2]
raw_data = {'surf': [70, 25, 1], 'elev': [30,75,5],'gaex': [0,0,22],'h2so4': [0,0,2],'so4': [0,0,70]}
df = pd.DataFrame(raw_data)

# From raw value to percentage
totals = [i+j+k+l+m for i,j,k,l,m in zip(df['surf'], df['elev'], df['gaex'], df['h2so4'], df['so4'])]
surf = [i / j * 100 for i,j in zip(df['surf'], totals)]
elev = [i / j * 100 for i,j in zip(df['elev'], totals)]
gaex = [i / j * 100 for i,j in zip(df['gaex'], totals)]
h2so4 = [i / j * 100 for i,j in zip(df['h2so4'], totals)]
so4 = [i / j * 100 for i,j in zip(df['so4'], totals)]

# plot
barWidth = 0.85
names = ('BC','POM','Sulfate')
fig = plt.figure(figsize=(16,6))
ax=plt.subplot(121)
plt.bar(r, surf, color='#377eb8', edgecolor='white', width=barWidth,label="Surf emis",zorder=4,alpha=0.8)
plt.bar(r, elev, bottom=surf, color='#ff7f00', edgecolor='white', width=barWidth,label="Elev emis",zorder=4,alpha=0.8)
plt.bar(r, gaex, bottom=[i+j for i,j in zip(surf, elev)], color='#4daf4a', edgecolor='white', width=barWidth,label="Gas-aero\nexchange",zorder=4,alpha=0.8)

plt.bar(r, h2so4, bottom=[i+j+k for i,j,k in zip(surf, elev,gaex)], color='#f781bf', edgecolor='white', width=barWidth,label='AQ chem\n(H$_2$SO$_4$)',zorder=4,alpha=0.8)
plt.bar(r, so4, bottom=[i+j+k+l for i,j,k,l in zip(surf, elev,gaex,h2so4)], color='#a65628', edgecolor='white', width=barWidth,label='AQ chem\n(SO$_4$)',zorder=4,alpha=0.8)
ax.grid(linestyle='--',color='#EBE7E0',zorder=3)
plt.xticks(r, names)
plt.ylim([-.5,100.5])
ax.set_xticks([0,1,2])
ax.set_xticklabels(('Black Carbon', 'Primary Organic \nMatter','Sulfate'),fontsize=12)
plt.setp(ax.spines.values(),lw=1.5)
plt.ylabel('Percent contribution (%)',fontsize=15)
ax.set_yticks([0,20,40,60,80,100])
ax.set_yticklabels(('0', '20','40', '60','80','100'),fontsize=12)
plt.rc('legend',fontsize=12)
ax.axvline(0.85, 1, 1.27, color='k', ls='-' , clip_on=False,lw=1.5)
leg = plt.legend(loc='upper left', bbox_to_anchor=(0.45,1.28), ncol=2,frameon=False)
ax.text(0.05,0.95,'(a)',size=15,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'},zorder=5)
ax.text(0.01,1.08,'Sources',size=15,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'},weight='bold')
ax.text(0.4,1.22,'Legend',rotation=90,size=15,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'},zorder=5)

r = [0,1,2]
raw_data = {'gvf': [2, 2, 4], 'tbf': [41,36,14],'sfin': [40,43,65], 'convin':[16,18,16], 'below':[1,1,1]}
df = pd.DataFrame(raw_data)

# From raw value to percentage
totals = [i+j+k+l+m for i,j,k,l,m in zip(df['gvf'], df['tbf'], df['sfin'], df['convin'], df['below'])]
surf = [i / j * 100 for i,j in zip(df['gvf'], totals)]
elev = [i / j * 100 for i,j in zip(df['tbf'], totals)]
gaex = [i / j * 100 for i,j in zip(df['sfin'], totals)]
convin = [i / j * 100 for i,j in zip(df['convin'], totals)]
below = [i / j * 100 for i,j in zip(df['below'], totals)]

# plot
barWidth = 0.85
names = ('BC','POM','Sulfate')
# Create green Bars
ax=plt.subplot(122)
plt.bar(r, surf, color='#e41a1c', edgecolor='white', width=barWidth,label="Dry dep (Grav)",zorder=4,alpha=0.8)
# Create orange Bars
plt.bar(r, elev, bottom=surf, color='#dede00', edgecolor='white', width=barWidth,label="Dry dep (Turb)",zorder=4,alpha=0.8)
# Create blue Bars
plt.bar(r, gaex, bottom=[i+j for i,j in zip(surf, elev)], color='#984ea3', edgecolor='white', width=barWidth,label="Wet dep\n(incloud, strat)",zorder=4,alpha=0.8)
plt.bar(r, convin, bottom=[i+j+k for i,j,k in zip(surf, elev,gaex)], color='k', edgecolor='white', width=barWidth,label="Wet dep\n(incloud, conv)",zorder=4,alpha=0.8)
plt.bar(r, below, bottom=[i+j+k+l for i,j,k,l in zip(surf, elev,gaex,convin)], color='#999999', edgecolor='white', width=barWidth,label="Wet dep\n(belowcloud)",zorder=4,alpha=0.8)

ax.grid(linestyle='--',color='#EBE7E0',zorder=3)
plt.xticks(r, names)
plt.ylim([-.5,100.5])
ax.set_xticks([0,1,2])
ax.set_xticklabels(('Black Carbon', 'Primary Organic \nMatter','Sulfate'),fontsize=12)
plt.setp(ax.spines.values(),lw=1.5)
plt.rc('legend',fontsize=12)
ax.set_yticks([0,20,40,60,80,100])
ax.set_yticklabels(('0', '20','40', '60','80','100'),fontsize=12)
ax.axvline(.4, 1, 1.27, color='k', ls='-' , clip_on=False,lw=1.5)
leg = plt.legend(loc='upper left', bbox_to_anchor=(0.3,1.28), ncol=2,frameon=False)
ax.text(0.05,0.95,'(b)',size=15,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'},zorder=5)
ax.text(0.01,1.08,'Sinks',size=15,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'},weight='bold')
ax.text(0.25,1.22,'Legend',rotation=90,size=15,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'},zorder=5)

# plt.savefig('fig09.pdf',dpi=300,format='pdf',bbox_inches='tight',pad_inches=0.1)
# plt.savefig('fig09.png',dpi=300,format='png',bbox_inches='tight',pad_inches=0.1)
###############################################################
## Figure 10
###############################################################
## Grouped barplot
barW = 0.25
bars1 = np.array([0.719,0.49,0.23])*100
bars2 = np.array([0.451,0.004,0.347,0.07,0.027,0.001])*100
bars3 = np.array([0.72,0.156,0.564])*100
bars4 = np.array([0.337,0.004,0.214,0.07,0.046,0.002])*100

x1 = np.arange(len(bars1))
x2 = np.arange(len(bars2))+0.5+len(bars1)
x3 = np.arange(len(bars3))+0.5+len(bars2)+0.5+len(bars1)
x4 = np.arange(len(bars4))+0.5+len(bars3)+0.5+len(bars2)+0.5+len(bars1)

ind = np.concatenate((x1,x2,x3,x4))

df_2 = pd.DataFrame({'snk':  ['Sources', 'Sources', 'Sources','Sinks','Sinks','Sinks','Sinks'],
             'Experiment': ['Total','Surf emis.','Elev emis.', 'Total','Grav','Turb','Wet'],
             'Result': [0.72,0.35,0.37,0.46,0.08,0.32,0.07]
             })

fig, ax = plt.subplots(figsize=(20, 7))
rects1 = ax.bar(x1, bars1, 0.8, color='#63ACBE',  edgecolor= "black",label="Sources",zorder=5)
for bar in rects1[1:]:
    print(bar.set_hatch('\\'))

rects2 = ax.bar(x2, bars2, 0.8, color='#EE442F',  edgecolor= "black",label="Sinks",zorder=5)
for bar in rects2[1:]:
    print(bar.set_hatch('\\'))

rects3 = ax.bar(x3, bars3, 0.8, color='#63ACBE',  edgecolor= "black",label="Sources",zorder=5)
for bar in rects3[1:]:
    print(bar.set_hatch('\\'))
    
rects4 = ax.bar(x4, bars4, 0.8, color='#EE442F',  edgecolor= "black",label="Sinks",zorder=5)
for bar in rects4[1:]:
    print(bar.set_hatch('\\'))
    
ax.set_ylabel('Normalized RMSE (%)',fontsize=14)
ax.set_xticks(ind)
ax.set_xticklabels(('Total', 'Surf\nemis','Elev\nemis', 'Total','Dry dep\n(Grav)','Dry dep\n(Turb)','Wdep\n(incloud\nstrat)','Wdep\n(incloud\nconv)','Wdep\n(below\ncloud)',\
                   'Total', 'Surf\nemis','Elev\nemis', 'Total','Dry dep\n(Grav)','Dry dep\n(Turb)','Wdep\n(incloud\nstrat)','Wdep\n(incloud\nconv)','Wdep\n(below\ncloud)'),fontsize=12)
ax.set_yticks([0,20,40,60,80])
ax.set_yticklabels(('0', '20','40', '60','80'),fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(linestyle='--',color='#EBE7E0',zorder=3)
        
#for pos in (machine_pos[:-1] + machine_pos[1:]) / 2:
ax.axvline(3-0.5/2, 0, -0.22, color='k', ls=':' , clip_on=False)
ax.axvline(10-1.5/2, 0, -0.22, color='k', ls=':' , clip_on=False)
ax.axvline(14-2.5/2, 0, -0.22, color='k', ls=':' , clip_on=False)

ax.text(0.05,-0.15,'BC Sources',size=14,transform=ax.transAxes,va='top',color='#63ACBE',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
ax.text(0.32,-0.15,'BC Sinks',size=14,transform=ax.transAxes,va='top',color='#EE442F',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
ax.text(0.56,-0.15,'POM Sources',size=14,transform=ax.transAxes,va='top',color='#63ACBE',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
ax.text(0.56+.25,-0.15,'POM Sinks',size=14,transform=ax.transAxes,va='top',color='#EE442F',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})

plt.xlim([-0.8,19])

plt.setp(ax.spines.values(),lw=1.5)

# plt.savefig('fig10.pdf',dpi=300,format='pdf',bbox_inches='tight',pad_inches=0.1)
# plt.savefig('fig10.png',dpi=300,format='png',bbox_inches='tight',pad_inches=0.1)
###############################################################
## Figure 11
###############################################################
## Grouped barplot
barW = 0.25
plt.rc('font', family='Helvetica')
bars1 = np.array([.36,.0037,0.05,0.086,0.01,.21])*100
bars2 = np.array([.09,0.007,0.012,0.06,0.014,0.0006])*100
x1 = np.arange(len(bars1))
x2 = np.arange(len(bars2))+1+len(bars1)
ind = np.concatenate((x1,x2))

df_2 = pd.DataFrame({'snk':  ['Sources', 'Sources', 'Sources','Sinks','Sinks','Sinks','Sinks'],
             'Experiment': ['Total','Surf emis.','Elev emis.', 'Total','Grav','Turb','Wet'],
             'Result': [0.72,0.33,0.38,0.35,0.07,0.2,0.07]
             })

fig, ax = plt.subplots(figsize=(16, 7))
rects1 = ax.bar(x1, bars1, 0.8, color='#63ACBE',  edgecolor= "black",label="Sources",zorder=5)
for bar in rects1[1:]:
    print(bar.set_hatch('\\'))

rects2 = ax.bar(x2, bars2, 0.8, color='#EE442F',  edgecolor= "black",label="Sinks",zorder=5)
for bar in rects2[1:]:
    print(bar.set_hatch('\\'))
    
ax.set_ylabel('Normalized RMSE (%)',fontsize=14)
ax.set_xticks(ind)
ax.set_xticklabels(('Total', 'Surf\nemis','Elev\nemis', 'Gas-aero\nexchange','AQ chem\n(H$_2$SO$_4$)','AQ chem\n(SO$_4$)','Total','Dry dep\n(Grav)','Dry dep\n(Turb)','Wdep\n(incloud\nstrat)','Wdep\n(incloud\nconv)','Wdep\n(below\ncloud)'),fontsize=12)
ax.set_yticks([0,10,20,30,40])
ax.set_yticklabels(('0', '10','20', '30','40'),fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(linestyle='--',color='#EBE7E0',zorder=3)
        
ax.axvline(6, 0, -0.22, color='k', ls=':' , clip_on=False)
ax.text(0.2,-0.15,'Sulfate Sources',size=14,transform=ax.transAxes,va='top',color='#63ACBE',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
ax.text(0.75,-0.15,'Sulfate Sinks',size=14,transform=ax.transAxes,va='top',color='#EE442F',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
plt.setp(ax.spines.values(),lw=1.5)

# plt.savefig('fig11.pdf',dpi=300,format='pdf',bbox_inches='tight',pad_inches=0.1)
# plt.savefig('fig11.png',dpi=300,format='png',bbox_inches='tight',pad_inches=0.1)

###############################################################
## Figure 13
###############################################################
## All available sites with emission mask
lab=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
lab=['(a)','(b)','(c)','(d)']
obs='obs'
varbs = ['bc_a1_SRF','SE_bc_a1_SRF','bc_a4_SRF','SE_bc_a4_SRF','bc','SE_bc']
df = pd.read_csv('Data_manuscript/BC_IMPROVE_dailydata.csv')
variable='BC'
df.index=pd.to_datetime(df['time'])
## Get rid of nans & -ve vals
df_nona = df[df['obs'].notna()]
df_nona = df_nona[df_nona['obs']>0]
res='RRM'
df_nona=get_emisMasked(df_nona,res,stat='diff',factor=0.68)
df_monthly = df_nona.groupby('ncols_'+res).resample('1M').mean()
df_annual = df_nona.groupby('ncols_'+res).resample('1Y').mean()
## Plotting
plt.figure(figsize=(22,12))
i=1
j=1
for v,treat,ylab,xlab in zip(varbs[-2:],['BC ($\u03BCg\ m^{-3}$)                  0.22',''],['RRM-PD\n\nModel','RRM-SE-PD\n\nModel'],['','Obs']):
    ax=plt.subplot(2,4,i)
    get_scatter_plot2(df_monthly,v,res,ax,treatment=treat,temp='monthly',size=5,cax=[5e-3,5e-3,3e0,3e0],vv=ylab,vx=xlab,typ=obs)
    ax.text(0.05,0.95,lab[j-1],size=20,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
    i+=4
    j+=2
## All available sites with emission mask
varbs = ['pom_a1_SRF','SE_pom_a1_SRF','pom_a4_SRF','SE_pom_a4_SRF','pom','SE_pom']
df = pd.read_csv('Data_manuscript/POM_IMPROVE_dailydata.csv')
variable='POM'
df.index=pd.to_datetime(df['time'])
## Get rid of nans & -ve vals
df_nona = df[df['obs'].notna()]
df_nona = df_nona[df_nona['obs']>0]
res='RRM'
## Masking data based on mean +/- 0.5*Standard deviation
df_nona=get_emisMasked(df_nona,res,stat='diff',factor=0.68)
## Estimating the monthly and annual averages from daily data
df_monthly = df_nona.groupby('ncols_'+res).resample('1M').mean()
df_annual = df_nona.groupby('ncols_'+res).resample('1Y').mean()
## Plotting
i=2
j=2
for v,treat,ylab,xlab in zip(varbs[-2:],['POM ($\u03BCg\ m^{-3}$)               1.12',''],['',''],['','Obs']):
    ax=plt.subplot(2,4,i)
    get_scatter_plot2(df_monthly,v,res,ax,treatment=treat,temp='monthly',size=5,cax=[1e-2,1e-2,5e1,5e1],vv=ylab,vx=xlab,typ=obs)
    ax.text(0.05,0.95,lab[j-1],size=20,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
    i+=4
    j+=2

# plt.savefig('fig13.png',format='png',dpi=300,bbox_inches='tight',pad_inches=0.1)
# plt.savefig('fig13.pdf',format='pdf',dpi=300,bbox_inches='tight',pad_inches=0.1)
###############################################################
## Figure 14
###############################################################
###############################################################
varbs = ['pom_a1_SRF','SE_pom_a1_SRF','pom_a4_SRF','SE_pom_a4_SRF','bc','SE_bc']
df = pd.read_csv('Data_manuscript/BC_IMPROVE_dailydata_new.csv')
variable='BC'
res='RRM'
obs='ECf_Val'
var=variable.lower()+'_'+res
try:
    df.index=pd.to_datetime(df['Date'])
except:
    df.index=pd.to_datetime(df['time'])

## Get rid of nans & -ve vals
df_nona = df[df[obs].notna()]
df_nona = df[df[varbs[-1]+'_'+res].notna()]
df_nona = df_nona[df_nona[obs]>0]

df_nona=get_emisMasked(df_nona,res,stat='diff',factor=0.68)
#df_nona2=get_ElevEmisMasked(df_nona,res,stat='diff',factor=0.68)
#df_nona=pd.concat([df_nona1,df_nona2])

df_nona['tt']=df_nona.index
df_nona['mon']=df_nona['tt'].dt.month
df_nona['season'] = df_nona['tt'].dt.month.map({
    12: 0, 1: 0, 2: 0,
    3: 1, 4: 1, 5: 1,
    6: 2, 7: 2, 8: 2, 9: 2,
    10: 3, 11: 3
})

df1=pd.DataFrame()
df1['BC']=df_nona['bc_'+res]
df1['mon'] = df_nona['mon']
df1['season'] = df_nona['season']
df1['Treatment']='Default treatment'
df1['YY']='ANN'
print(df1['BC'].var())

df2=pd.DataFrame()
df2['BC']=df_nona['SE_bc_'+res]
df2['mon'] = df_nona['mon']
df2['season'] = df_nona['season']
df2['Treatment']='Improved treatment'
df2['YY']='ANN'
print(df2['BC'].var())

df3=pd.DataFrame()
df3['BC']=df_nona[obs]
df3['mon'] = df_nona['mon']
df3['season'] = df_nona['season']
df3['Treatment']='Observation'
df3['YY']='ANN'
print(df1['BC'].var())

fdf = pd.concat([df1,df2,df3])

fig=plt.figure(figsize=[16,8])
ax1=plt.subplot(121)
ax1=sns.boxplot(x="season",y='BC',data=fdf,hue="Treatment",palette='Set3',showfliers=False,\
            showmeans=True,zorder=3,\
            meanprops={"marker":"d","markerfacecolor":"red", "markeredgecolor":"blue"})
ax1.grid(linestyle='--',color='#EBE7E0',zorder=4)
ax1.set_axisbelow(True)
plt.ylabel('Concentration ($\u03BC$g m$^{-3}$)',fontsize=20)
plt.xlabel('Seasons',fontsize=15)
plt.tick_params(labelsize=15)
plt.legend([],[], frameon=False)
plt.setp(ax1.spines.values(),lw=1.5)
plt.xticks([0,1,2,3],['DJF','MAM','JJA','SON'])
ax1.text(0.05,0.95,'(a)',size=20,transform=ax1.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
ax1.text(0.01,1.05,'Surface BC',size=20,transform=ax1.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})

varbs = ['pom_a1_SRF','SE_pom_a1_SRF','pom_a4_SRF','SE_pom_a4_SRF','pom','SE_pom']
df = pd.read_csv('Data_manuscript/POM_IMPROVE_dailydata_new.csv')
variable='POM'
res='RRM'
obs='OCf_Val'
var=variable.lower()+'_'+res
try:
    df.index=pd.to_datetime(df['time'])
except:
    df.index=pd.to_datetime(df['Date'])

## Get rid of nans & -ve vals
df_nona = df[df[obs].notna()]
df_nona = df[df['pom_RRM'].notna()]
df_nona = df[df['SE_pom_RRM'].notna()]
df_nona = df_nona[df_nona[obs]>0]

df_nona=get_emisMasked(df_nona,res,stat='diff',factor=0.68)

df_nona['tt']=df_nona.index
df_nona['mon']=df_nona['tt'].dt.month
df_nona['season'] = df_nona['tt'].dt.month.map({
    12: 0, 1: 0, 2: 0,
    3: 1, 4: 1, 5: 1,
    6: 2, 7: 2, 8: 2, 9: 2,
    10: 3, 11: 3
})

df1=pd.DataFrame()
df1['pom']=df_nona['pom_RRM']
df1['mon'] = df_nona['mon']
df1['season'] = df_nona['season']
df1['treat']='RRM-PD'
df1['YY']='ANN'

df2=pd.DataFrame()
df2['pom']=df_nona['SE_pom_RRM']
df2['mon'] = df_nona['mon']
df2['season'] = df_nona['season']
df2['treat']='RRM-SE-PD'
df2['YY']='ANN'

df3=pd.DataFrame()
df3['pom']=df_nona[obs]
df3['mon'] = df_nona['mon']
df3['season'] = df_nona['season']
df3['treat']='Observation'
df3['YY']='ANN'

fdf = pd.concat([df1,df2,df3])

#plt.figure(figsize=[8,8])
ax1=plt.subplot(122)
ax1 = sns.boxplot(x="season",y='pom',data=fdf,hue="treat",palette='Set3',showfliers=False,\
            showmeans=True,zorder=3,\
            meanprops={"marker":"d","markerfacecolor":"red", "markeredgecolor":"blue"})
ax1.grid(linestyle='--',color='#EBE7E0',zorder=3)
ax1.set_axisbelow(True)
plt.ylabel('',fontsize=15)
plt.xlabel('Seasons',fontsize=15)
plt.tick_params(labelsize=15)
plt.legend(fontsize=15,loc='upper center')
plt.setp(ax1.spines.values(),lw=1.5)
plt.xticks([0,1,2,3],['DJF','MAM','JJA','SON'])
ax1.text(0.05,0.95,'(b)',size=20,transform=ax1.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
ax1.text(0.01,1.05,'Surface POM',size=20,transform=ax1.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
ax1.set_ylim([-.2,5.4])
fig.subplots_adjust(right=0.95)

# plt.savefig('fig14.png',format='png',dpi=300,bbox_inches='tight',pad_inches=0.1)
# plt.savefig('fig14.pdf',format='pdf',dpi=300,bbox_inches='tight',pad_inches=0.1)
###################
###############################################################
## Figure 15
###############################################################
##################
varbs = ['so4_a1_SRF','SE_so4_a1_SRF','so4_a2_SRF','SE_so4_a2_SRF','so4','SE_so4']
df = pd.read_csv('SO4_IMPROVE_dailydata.csv')
df[['so4_RRM','SE_so4_RRM']] = df[['so4_RRM','SE_so4_RRM']]*(96/115)
variable='SO$_4$'
df.index=pd.to_datetime(df['time'])
## Get rid of nans & -ve vals
df_nona = df[df['obs'].notna()]
df_nona = df_nona[df_nona['obs']>0]
obs='obs'
res='RRM'
## Masking data based on mean +/- 0.5*Standard deviation
df_nona=get_emisMasked(df_nona,res,stat='diff',factor=0.68)
## Estimating the monthly and annual averages from daily data
df_monthly = df_nona.groupby('ncols_'+res).resample('1M').mean()
df_annual = df_nona.groupby('ncols_'+res).resample('1Y').mean()
## Plotting
plt.figure(figsize=(22,12))
i=1
j=1
for v,treat,ylab,xlab in zip(varbs[-2:],['SO$_4$ ($\u03BCg\ m^{-3}$)                0.74',''],['RRM-PD\n\nModel','RRM-SE-PD\n\nModel'],['','Obs']):
    ax=plt.subplot(2,4,i)
    get_scatter_plot2(df_monthly,v,res,ax,treatment=treat,temp='monthly',size=5,cax=[5e-2,5e-2,1e1,1e1],vv=ylab,vx=xlab,typ=obs)
    ax.text(0.05,0.95,lab[j-1],size=20,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
    i+=4
    j+=2
#########
## Mask data based on emissions
def get_emisMasked2(df,res,var,stat='diff',factor=1,pos=None,neg=None):
    print(stat)
    if pos==None:
        pos=df[var+'_emis_'+stat+'_'+res].mean()+factor*df[var+'_emis_'+stat+'_'+res].std()
    if neg==None:
        neg=df[var+'_emis_'+stat+'_'+res].mean()-factor*df[var+'_emis_'+stat+'_'+res].std()
    df=df[(df[var+'_emis_'+stat+'_'+res]<=neg) | (df[var+'_emis_'+stat+'_'+res]>=pos)] 
    print(pos,neg)
    return df

## Mask data based on emissions
def get_ElevEmisMasked2(df,res,var,stat='diff',factor=1,pos=None,neg=None):
    print(stat)
    if pos==None:
        pos=df[var+'_emis_elev_'+stat+'_'+res].mean()+factor*df[var+'_emis_elev_'+stat+'_'+res].std()
    if neg==None:
        neg=df[var+'_emis_elev_'+stat+'_'+res].mean()-factor*df[var+'_emis_elev_'+stat+'_'+res].std()
    print('Range: ',pos,neg)
    df=df[(df[var+'_emis_elev_'+stat+'_'+res]<=neg) | (df[var+'_emis_elev_'+stat+'_'+res]>=pos)] 
    return df

obs='AOD_500nm'
varbs = ['AODVIS','SE_AODVIS','angstrm','SE_angstrm','AODABS','AODBC','AODPOM','AODSO4']
df = pd.read_csv('AERONET_modelAOD_ALL.csv')
df.index=pd.to_datetime(df['time'])
variable=varbs[0]
df.index=pd.to_datetime(df['time'])
## Get rid of nans & -ve vals
df_nona = df[df[obs].notna()]
df_nona = df_nona[df_nona[obs]>=0]
df_nona = df_nona[df_nona[obs]<=1]
res='RRM'
## Masking sites with high bc and pom emissions difference
df_nona1=get_emisMasked2(df_nona,res,'bc_a4',stat='diff',factor=0.68)
df_nona2=get_ElevEmisMasked2(df_nona,res,'bc_a4',stat='diff',factor=0.68)
df_nona3=get_emisMasked2(df_nona,res,'pom_a4',stat='diff',factor=0.68)
df_nona4=get_ElevEmisMasked2(df_nona,res,'pom_a4',stat='diff',factor=0.68)
df_nona=pd.concat([df_nona1,df_nona3,df_nona2,df_nona4])

df_monthly = df_nona.groupby(res).resample('1M').mean()
df_annual = df_nona.groupby(res).resample('1Y').mean()
## Plotting
i=2
j=2
for v,treat,ylab,xlab in zip(varbs[:2],['AOD ($1$)                         0.11',''],['',''],['','Obs']):
    ax=plt.subplot(2,4,i)
    get_scatter_plot3(df_monthly,v,res,ax,treatment=treat,temp='monthly',size=5,cax=[1e-2,1e-2,1e0,1e0],vv=ylab,vx=xlab,typ=obs)
    ax.text(0.05,0.95,lab[j-1],size=20,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
    i+=4
    j+=2
    
# plt.savefig('fig15.png',format='png',dpi=300,bbox_inches='tight',pad_inches=0.1)
# plt.savefig('fig15.pdf',format='pdf',dpi=300,bbox_inches='tight',pad_inches=0.1)




