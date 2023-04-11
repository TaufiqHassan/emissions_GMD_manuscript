#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 15:55:51 2023

@author: hass877
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib.lines as mlines
from matplotlib.patches import PathPatch
from matplotlib import collections

def get_crange(v1,v2):
    aagg = (np.max(v1.values)+np.max(v2.values))/2
    aagg = np.log10(aagg)
    expo = np.floor(aagg)
    bbgg = aagg - expo
    if 10**(bbgg)<2.:
        s1 = [5*10**(expo-4),1*10**(expo-3),2*10**(expo-3), \
                5*10**(expo-3),1*10**(expo-2),2*10**(expo-2), \
                5*10**(expo-2),1*10**(expo-1),2*10**(expo-1), \
                5*10**(expo-1),10**expo,      2.*10**expo]
    elif 10**(bbgg)<5.:
        s1 = [1*10**(expo-3),2*10**(expo-3),5*10**(expo-3), \
                1*10**(expo-2),2*10**(expo-2),5*10**(expo-2), \
                1*10**(expo-1),2*10**(expo-1),5*10**(expo-1), \
                10**expo,      2.*10**expo,   5.*10**expo]
    else:
        s1 = [2*10**(expo-3),5*10**(expo-3),1*10**(expo-2), \
                2*10**(expo-2),5*10**(expo-2),1*10**(expo-1), \
                2*10**(expo-1),5*10**(expo-1),10**expo,       \
                2.*10**expo,   5.*10**expo,   10**(expo+1)]
    return s1
def get_crange2(diff):
    aagg = np.max(abs(diff).values)
    aagg = np.log10(aagg)
    expo = np.ceil(aagg)
    s2 = np.array([-100,-50,-20,-10,-5,-2,-1,1,2,5,10,20,50,100])*(10**(expo)/1000)
    return s2

def get_lines(ds,ax):
    x = ds['coord'][0,:].squeeze()
    y = ds['coord'][1,:].squeeze()
    z = ds['coord'][2,:].squeeze()
    lon = np.arctan2(y, x) * 180.0 / np.pi
    lat = np.arcsin(z) * 180.0 / np.pi
    corner_indices = ds['connect1']
    xx = lon[corner_indices[:,:] - 1]
    yy = lat[corner_indices[:,:] - 1]
    lines = [[[xx[i,j], yy[i,j]] for j in range(xx.shape[1])] for i in range(xx.shape[0])]
    line_collection = collections.LineCollection(lines, transform=crs.Geodetic(),colors='k',linewidth=0.5)
    ax.add_collection(line_collection)

def rounding(n):
    if (type(n)==str) or (np.isnan(n)):
        return str('-')
    elif ((abs(n)>1e-5) and (abs(n)<1e5)):
        try:
            sgn = '-' if n<0 else ''
            num = format(abs(n)-int(abs(n)),'f')
            if int(num[2:])<1:
                d = str((abs(n)))
                return sgn + d
            else:
                for i,e in enumerate(num[2:]):
                    if e!= '0':
                        if i==0:
                            d = str(int(abs(n))) + (num[1:i+5])
                        else:
                            d = str(int(abs(n))) + (num[1:i+4])
                        return sgn+d
        except:
            return '-'
    else:
        return '{:.0e}'.format(n)

def get_stat(var2,var,weight=1):
    land = xr.open_dataset('LANDFRAC_RRM.nc')['LANDFRAC'][0]
    area = xr.open_dataset('total_bb_2014_surf_625_384x576_northamericax4v1pg2.nc')['area']
    area.lon[area.lon > 180.] -= 360.
    garea = area.copy()
    garea = garea.where(garea.lat>=15).where(garea.lat<=75).where(garea.lon<=-50).where(garea.lon>=-170).where(land>0).dropna(dim='ncol')
    var = var.where(var.lat>=15).where(var.lat<=75).where(var.lon<=-50).where(var.lon>=-170).where(land>0).dropna(dim='ncol')
    var2 = var2.where(var2.lat>=15).where(var2.lat<=75).where(var2.lon<=-50).where(var2.lon>=-170).where(land>0).dropna(dim='ncol') 
    mean = (var2*garea).sum(garea.dims)/(garea).sum(garea.dims)
    RMSE=np.sqrt(np.divide(np.sum(np.multiply((var2-var),(var2-var))),len(var)))
    WRMSE=np.sqrt(np.divide(np.sum(np.multiply(weight*(var2-var),weight*(var2-var))),len(var)))
    NRMSE = (RMSE/np.std(var))
    WNRMSE = (WRMSE/np.std(var))
    dd = pd.DataFrame([(mean),(RMSE.values),(WRMSE.values),(NRMSE.values),(WNRMSE.values)]).T
    dd.columns = ['Mean (accurate)', 'RMSE', 'WRMSE', 'NRMSE', 'WNRMSE']
    return mean.values, RMSE.values, NRMSE.values

def get_nearestlatlon(lon1,lat1,lon,lat):
    try:
        ind=np.argmin((lon-lon1)**2+(lat.values-lat1)**2)
        lat1,lat2,lon1,lon2 = lat[ind],lat[ind],lon[ind],lon[ind]
    except:
        RLLlon = lon.sel(lon=lon1, method='nearest')
        RLLlat = lat.sel(lat=lat1, method='nearest')
        lat1,lat2,lon1,lon2 = RLLlat,RLLlat,RLLlon,RLLlon
    return lat1,lat2,lon1,lon2,ind

def get_mod_alt(mdata,ind,alt):
    z3 = xr.open_dataset('Z3_RRM.nc')['Z3']
    ndata = np.empty((len(mdata.time),len(alt)))
    for i in range(len(mdata.time)):
        ndata[i,:] = np.interp(alt,np.flip(z3[:,ind]),np.flip(mdata[i,:,ind]),left=np.nan)
    new_data = xr.DataArray(data=ndata,  dims=["time","altitude"],
        coords=dict(altitude=(["altitude"], (alt).astype(float)), time=(["time"], mdata.time.data),\
                    ))
    return new_data

def getVmap_alt(data,ranges,ax,unit,cm,alt,fig):
    alind = pd.DataFrame(data[0,:]).dropna().index[0]
    x,y = np.meshgrid(data['altitude'][alind:]-alt[alind],np.arange(len(data['time'])))
    print(x.shape,y.shape)
    im=ax.contourf(y,x,data[:][:,alind:],cmap=cm,levels=ranges,norm=matplotlib.colors.BoundaryNorm(boundaries=ranges, ncolors=236))
    ax.contour(y,x,data[:][:,alind:],levels=ranges,linestyles='--',alpha=0.3)
    plt.yticks([100,300,500,1000,2000,4000])
    plt.ylim([0,4000])
    plt.xlim([0,30])
    ax.yaxis.set_tick_params(width=1.5,length=5)
    ax.xaxis.set_tick_params(width=1.5,length=5)
    ax.grid( lw=0.5, color='#EBE7E0', alpha=0.5, linestyle='-.')
    ax.tick_params(labelsize=12,labelbottom=False)
    positions = ax.get_position()
    cax = fig.add_axes([positions.x1+0.01,positions.y0,0.02,positions.y1-positions.y0])
    cbar=plt.colorbar(im,cax=cax,ticks=ranges,drawedges=True)
    s1 = pd.DataFrame(ranges)
    s2 = s1.applymap(lambda x: rounding(x) if ((abs(x)>1e-5) and (abs(x)<1e5)) else '{:.0e}'.format(x))[0].tolist()
    cbar_ticks_old=list(map(str,s2))
    cbar_ticks = []
    for i in cbar_ticks_old:
        if '.' in i:
            cbar_ticks.append(i.rstrip('0').rstrip('.'))
        else:
            cbar_ticks.append(i)
    #cbar_ticks = [i.rstrip('0').rstrip('.') for i in cbar_ticks]
    cbar.ax.set_yticklabels(['']+cbar_ticks[1:-1]+[''],size=12)
    cbar.set_label(label=unit,size=12)
    cbar.outline.set_linewidth(1.5)
    cbar.dividers.set_linewidth(1.5)
    plt.setp(ax.spines.values(),lw=1.5)
    #ax.set_ylabel('Altitude [m]',fontsize=15)

def get_ts(data,ax2,unit,ind):
    #alind = pd.DataFrame(data[0,:]).dropna().index[0]
    #data = data[:,alind:]
    ax2.plot(data[:,ind],c='#984ea3',zorder=1)
    #ax2.set_ylim([-100,100])
    ax2.yaxis.set_tick_params(width=1.5,length=5)
    ax2.xaxis.set_tick_params(width=1.5,length=5)
    ax2.grid( lw=0.5, color='#EBE7E0', alpha=0.5, linestyle='-.')
    ax2.tick_params(labelsize=12)
    plt.setp(ax2.spines.values(),lw=1.5)
    ax2.set_xlim([0,30])
    ax2.set_xticks([4,9,14,19,24,29])
    ax2.set_xticklabels(['5','10','15','20','25','30'],size=12)
    ax2.set_xlabel('Time [day]',fontsize=15)
    #ax2.set_ylabel('Burden\n'+unit,fontsize=15)
    ax2.axhline(linewidth=0.5,linestyle='--', color='darkgray',zorder=2)

def get_vertint(vdata,ha,p0,hb,ps,grav,fact):
    ## calc. dp
    delp = 0*vdata
    p = ha*p0+hb*ps
    if 'ncol' in p.dims:
        p = p.transpose('ilev','ncol')
    else:
        p = p.transpose('ilev','lat','lon')
    delp = p[1:,:].values-p[:-1,:].values
    delp = delp + 0*vdata
    ## unit conversion and vertical integration
    vdata = vdata*(delp/grav) # p/g = Pa/ms^-2 = Nm^-2/ms^-2 = Kg.ms^-2/m^2/ms^-2
    vdata = vdata*fact
    vdata = vdata.sum('lev')
    return vdata

def get_var(data,var,reg='SGP'):
    regions = {'SGP':'ncol_260e_to_265e_34n_to_39n',\
          'NSA':'ncol_201e_to_206e_69n_to_74n',\
          'TCAP':'ncol_287e_to_292e_40n_to_45n'}
    vv = data[var+'_'+regions[reg][5:]]
    lat = data['lat_'+regions[reg][5:]][0]
    lon = data['lon_'+regions[reg][5:]][0].values
    lon[lon > 180.] -= 360.
    newlon = xr.DataArray(lon,coords={'ncol_'+regions[reg][5:]:data['ncol_'+regions[reg][5:]].values})
    newlat = xr.DataArray(lat,coords={'ncol_'+regions[reg][5:]:data['ncol_'+regions[reg][5:]].values})
    vv = vv.assign_coords(lon=newlon)
    vv = vv.assign_coords(lat=newlat)
    vv=vv.rename({'ncol_'+regions[reg][5:]:'ncol'})
    vv.name=var
    vdata = xr.open_dataset('Data_manuscript/F20TR_v2_ndg_ERA5_SEdata_NA_RRM_CDT.eam.h1.2016-01-01-00000.nc')
    vlon = vdata['lon'].values
    vlon[vlon > 180.] -= 360.
    clats=(np.where(np.in1d(vdata.lat.values, newlat.values))[0])
    clons=(np.where(np.in1d(vlon, newlon.values))[0])
    inds=(clats[np.in1d(clats, clons)])
    return vv,inds

## Mask data based on emissions
def get_emisMasked(df,res,stat='diff',factor=1,pos=None,neg=None):
    if pos==None:
        pos=df['emis_'+stat+'_'+res].mean()+factor*df['emis_'+stat+'_'+res].std()
    if neg==None:
        neg=df['emis_'+stat+'_'+res].mean()-factor*df['emis_'+stat+'_'+res].std()
    df=df[(df['emis_'+stat+'_'+res]<=neg) | (df['emis_'+stat+'_'+res]>=pos)] 
    print(pos,neg)
    return df

## Mask data based on emissions
def get_ElevEmisMasked(df,res,stat='diff',factor=1,pos=None,neg=None):
    if pos==None:
        pos=df['emis_elev_'+stat+'_'+res].mean()+factor*df['emis_elev_'+stat+'_'+res].std()
    if neg==None:
        neg=df['emis_elev_'+stat+'_'+res].mean()-factor*df['emis_elev_'+stat+'_'+res].std()
    print('Range: ',pos,neg)
    df=df[(df['emis_elev_'+stat+'_'+res]<=neg) | (df['emis_elev_'+stat+'_'+res]>=pos)] 
    return df

## Get Scatter plots
def get_scatter_plot2(df_nona,v,res,ax,treatment='Default',temp='daily',size=1,cax=None,maxv=None,vv='aerosol Conc.',typ='obs',vx=None):
    var = v+'_'+res
    df_nona = df_nona[np.isfinite(df_nona[var])]
    df_nona = df_nona[df_nona[var].notna()].reset_index(drop=True)
    meanB = df_nona[var].mean() - df_nona[typ].mean()
    print(meanB)
    # Calculate the point density
    corr=stats.spearmanr(df_nona['obs'],df_nona[var])
    xy = np.vstack([df_nona[typ],df_nona[var]])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    improve_obs, mod, z_mod = df_nona[typ][idx], df_nona[var][idx], z[idx]
    if maxv==None:
        maxv = z_mod.max()
    #improve_obs, mod, z_mod = density_scatter(df['obs'], df['mod'])
    RMSE=np.sqrt(np.divide(np.sum(np.multiply((df_nona[var]-df_nona[typ]),(df_nona[var]-df_nona[typ]))),len(df_nona[typ])))
    NRMSE=(RMSE/(df_nona[typ].std()))*100
    meanB = abs(((df_nona[var] - df_nona[typ]).sum()/df_nona[typ].sum())*100)
    heatmap = ax.scatter(improve_obs, mod, c='k', s=size, edgecolor='k',vmax=maxv, vmin=0,zorder=2)
    plt.grid(linestyle='--',color='#EBE7E0',zorder=1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    print('Mean:',df_nona[typ].mean())
    ax.text(0.005,1.03,treatment,size=20,transform=ax.transAxes)
    ax.text(0.63,0.28,'R: '+str(np.round(corr[0],2)),size=12,transform=ax.transAxes)
    ax.text(0.63,0.22,'RMSE: '+str(np.round(RMSE,2)),size=12,transform=ax.transAxes)
    ax.text(0.63,0.16,'NMB: '+str(round(meanB))+'%',size=12,transform=ax.transAxes)
    #ax.text(0.63,0.1,'Mean: '+str(np.round(df_nona[typ].mean(),2)),size=12,transform=ax.transAxes)
    ax.text(0.63,0.1,'n: '+str(len(df_nona)),size=12,transform=ax.transAxes)
    plt.setp(ax.spines.values(),lw=1.5)
    if cax==None:
        if 'bc' in v:
            xx1=4e-4 # BC
            yy1=1e-3 # BC
        elif 'so4' in v:
            xx1=1e-2 # SO4
            yy1=1e-1 # SO4
        elif 'pom' in v:
            xx1=1e-2 # SO4
            yy1=1e-2 # SO4
        xx2=1e1
        yy2=1e1 # BC
    else:
        xx1,yy1,xx2,yy2=cax
    plt.xlim([xx1,xx2]) # BC
    plt.ylim([yy1,yy2]) #BC
    ###########
    x1 = mlines.Line2D([xx1, xx2], [yy1, yy2], color='k', linestyle='-',linewidth=1)
    ax.add_line(x1)
    x1 = mlines.Line2D([2*xx1, xx2], [yy1, 0.5*yy2], color='k', linestyle='--',linewidth=1)
    ax.add_line(x1)
    x1 = mlines.Line2D([xx1, 0.5*xx2], [2*yy1, yy2], color='k', linestyle='--',linewidth=1)
    ax.add_line(x1)
    ax.tick_params(labelsize=12)
    plt.ylabel(vv,fontsize=20)
    plt.xlabel(vx,fontsize=20)
    
def get_scatter_plot3(df_nona,v,res,ax,treatment='Default',temp='daily',size=1,cax=None,maxv=None,vv='aerosol Conc.',typ='obs',vx=None):
    var = v+'_'+res
    df_nona = df_nona[np.isfinite(df_nona[var])]
    df_nona = df_nona[df_nona[var].notna()].reset_index(drop=True)
    meanB = df_nona[var].mean() - df_nona[typ].mean()
    print(meanB)
    # Calculate the point density
    corr=stats.pearsonr(df_nona[typ],df_nona[var])
    xy = np.vstack([df_nona[typ],df_nona[var]])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    improve_obs, mod, z_mod = df_nona[typ][idx], df_nona[var][idx], z[idx]
    if maxv==None:
        maxv = z_mod.max()
    #improve_obs, mod, z_mod = density_scatter(df['obs'], df['mod'])
    RMSE=np.sqrt(np.divide(np.sum(np.multiply((df_nona[var]-df_nona[typ]),(df_nona[var]-df_nona[typ]))),len(df_nona[typ])))
    meanB = ((df_nona[var] - df_nona[typ]).sum()/df_nona[typ].sum())*100
    heatmap = ax.scatter(improve_obs, mod, c='k', s=size, edgecolor='k',vmax=maxv, vmin=0,zorder=2)
    plt.grid(linestyle='--',color='#EBE7E0',zorder=1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    #plt.title(treatment,size=20)
    print('mean:',df_nona[typ].mean())
    ax.text(0.005,1.03,treatment,size=20,transform=ax.transAxes)
    ax.text(0.63,0.28,'R: '+str(np.round(corr[0],3)),size=12,transform=ax.transAxes)
    ax.text(0.63,0.22,'RMSE: '+str(np.round(RMSE,3)),size=12,transform=ax.transAxes)
    ax.text(0.63,0.16,'NMB: '+str(np.round(meanB,1))+'%',size=12,transform=ax.transAxes)
    ax.text(0.63,0.1,'n: '+str(len(df_nona)),size=12,transform=ax.transAxes)
    #plt.colorbar(heatmap,label='Density')
    plt.setp(ax.spines.values(),lw=1.5)
    if cax==None:
        if 'bc' in v:
            xx1=4e-4 # BC
            yy1=1e-3 # BC
        elif 'so4' in v:
            xx1=1e-2 # SO4
            yy1=1e-1 # SO4
        elif 'pom' in v:
            xx1=1e-2 # SO4
            yy1=1e-2 # SO4
        xx2=1e1
        yy2=1e1 # BC
    else:
        xx1,yy1,xx2,yy2=cax
    plt.xlim([xx1,xx2]) # BC
    plt.ylim([yy1,yy2]) #BC
    ###########
    x1 = mlines.Line2D([xx1, xx2], [yy1, yy2], color='k', linestyle='-',linewidth=1)
    ax.add_line(x1)
    x1 = mlines.Line2D([2*xx1, xx2], [yy1, 0.5*yy2], color='k', linestyle='--',linewidth=1)
    ax.add_line(x1)
    x1 = mlines.Line2D([xx1, 0.5*xx2], [2*yy1, yy2], color='k', linestyle='--',linewidth=1)
    ax.add_line(x1)
    ax.tick_params(labelsize=12)
    plt.ylabel(vv,fontsize=20)
    plt.xlabel(vx,fontsize=20)

class get_plots(object):
    
    def __init__(self,var,ax,**kwargs):
        self.var = var
        self.ax = ax
        self.xint = kwargs.get('xint',None)
        self.yint = kwargs.get('yint',None)
        self.figsize = kwargs.get('figsize',None)
        self.scrip_file = kwargs.get('scrip_file',None)
        self.lat_range = kwargs.get('lat_range',[-90,90])
        self.lon_range = kwargs.get('lon_range',[-180,180])
        self.cm = kwargs.get('cmap',plt.cm.jet)
        self.labelsize = kwargs.get('labelsize',13)
        self.unit = kwargs.get('unit','unit')
        self.gridLines = kwargs.get('gridLines',True)
        self.colbar = kwargs.get('colbar',True)
        self.states = kwargs.get('states',False)
        self.map_proj = kwargs.get('projection',crs.PlateCarree())
        self.res = kwargs.get('res','110m')
        self.cbs = kwargs.get('cbs',0)
        self.cbe = kwargs.get('cbe',-1)
        self.cbi = kwargs.get('cbi',1)
        self.bdist = kwargs.get('bdist',0.06)
        self.th = kwargs.get('th',0.02)
        self.trot = kwargs.get('trot',0)
        self.hatches = kwargs.get('hatches',False)
        self.rr = kwargs.get('levels',[0.,0.000274,0.00307,0.0214,0.0793,.198,.392,.682,1.13,5.,10.,32.9])
    
        
    def get_verts(self):
        ds_scrip=xr.open_dataset(self.scrip_file)
        corner_lon = np.copy( ds_scrip.grid_corner_lon.values )
        corner_lat = np.copy( ds_scrip.grid_corner_lat.values )
        center_lon = np.copy( ds_scrip.grid_center_lon.values )
        if ((np.min(self.lon_range) < 0) & (np.max(corner_lon) > 180)):
            corner_lon[corner_lon > 180.] -= 360.
        
        lons_corners = np.copy(corner_lon.reshape(corner_lon.shape[0],corner_lon.shape[1],1))
        lats_corners = np.copy(corner_lat.reshape(corner_lat.shape[0],corner_lat.shape[1],1))
        lons_corners[lons_corners > 180.] -= 360
        center_lon[center_lon > 180.] -= 360
        
        new_lons_corners = []
        new_lats_corners = []
        varbl = []
        lon_maxmin = np.max(lons_corners,axis=(1,2)) - np.min(lons_corners,axis=(1,2))
        g180 = np.where(lon_maxmin>180)[0]
        g180l0 = np.where(np.mean(lons_corners[g180],axis=(1,2)) <= 0)[0]
        tmp_lons_corners = lons_corners[g180[g180l0]].copy()
        tmp_lons_corners = np.where(lons_corners[g180[g180l0]]<0,180,tmp_lons_corners)
        new_lons_corners.append(tmp_lons_corners)
        new_lats_corners.append(lats_corners[g180[g180l0]])
        lons_corners[g180[g180l0]] = np.where(lons_corners[g180[g180l0]]>0,-180,lons_corners[g180[g180l0]])
        varbl.append(self.var[g180[g180l0]])
        
        g180g0 = np.where(np.mean(lons_corners[g180],axis=(1,2)) > 0)[0]
        tmp_lons_corners = lons_corners[g180[g180g0]].copy()
        tmp_lons_corners = np.where(lons_corners[g180[g180g0]]>0,-180,tmp_lons_corners)
        new_lons_corners.append(tmp_lons_corners)
        new_lats_corners.append(lats_corners[g180[g180g0]])
        lons_corners[g180[g180g0]] = np.where(lons_corners[g180[g180g0]]<0,180,lons_corners[g180[g180g0]])
        varbl.append(self.var[g180[g180g0]])

        lons_corners = np.concatenate((lons_corners, np.array(new_lons_corners[0])), axis=0)
        lats_corners = np.concatenate((lats_corners, np.array(new_lats_corners[0])), axis=0)
        self.var = np.concatenate((self.var, np.array(varbl[0])), axis=0)        
        verts = np.concatenate((lons_corners, lats_corners), axis=2)
        return self.var, verts
        
    def get_map(self):
        kwd_polycollection = {}
        kwd_pcolormesh = {}
        if self.gridLines == True:
            kwd_polycollection['edgecolor'] = 'k'
            kwd_polycollection['lw'] = 0.05
            kwd_pcolormesh['edgecolors'] = 'k'
            kwd_pcolormesh['lw'] = 0.001
        #else:
        #    kwd_polycollection['edgecolor'] = 'face'
        #plt.rcParams['font.family'] = 'STIXGeneral'
        #plt.rcParams['font.family'] = 'Helvetica'
        plt.rc('font', family='Helvetica')
        ## levels
        ranges=self.rr
        self.ax.set_global()
        clen=len(np.arange(0,257)[self.cbs:self.cbe:self.cbi])
        try:
            self.cm = ListedColormap(self.cm.colors[self.cbs:self.cbe:self.cbi])
        except:
            self.cm = self.cm
            print('Cannot subscript Segmented Colormap!')
        if ('.nc' in str(self.scrip_file)) | (type(self.scrip_file)==int):
            var, verts = self.get_verts()
            im = PolyCollection(verts,cmap=self.cm,**kwd_polycollection,\
                               norm=matplotlib.colors.BoundaryNorm(boundaries=ranges, ncolors=clen) )
            im.set_array(var)
            if self.hatches != False:
                for path in im.get_paths():
                    patch = PathPatch(path,hatch='//',facecolor='none',edgecolor='k')
                    self.ax.add_patch(patch)
            self.ax.add_collection(im)
        else:
            try:
                lon = self.var.lon
                lat = self.var.lat
            except:
                lon = self.var.longitude
                lat = self.var.latitude
            im = self.ax.pcolormesh(lon, lat, self.var, cmap=self.cm, transform=self.map_proj, \
                                    **kwd_pcolormesh, norm=matplotlib.colors.BoundaryNorm(boundaries=ranges, ncolors=clen) )
        
        self.ax.set_xlim(self.lon_range)
        if self.xint == None:
            self.xint = np.around((self.lon_range[1]-self.lon_range[0])/6)
        xticklabels=np.arange(self.lon_range[0],self.lon_range[1]+self.xint,self.xint)
        self.ax.set_ylim(self.lat_range)
        if self.yint == None:
            self.yint = np.around((self.lat_range[1]-self.lat_range[0])/6)
        yticklabels=np.arange(self.lat_range[0],self.lat_range[1]+self.yint,self.yint)
        self.ax.coastlines(resolution=self.res,lw=0.5,edgecolor='k')
        self.ax.add_feature(cfeature.BORDERS.with_scale(self.res),lw=0.5,edgecolor='k')
        if self.states==True:
            self.ax.add_feature(cfeature.STATES.with_scale(self.res),lw=0.5,edgecolor='k')
        self.ax.set_xticks(xticklabels,crs=self.map_proj)
        self.ax.set_yticks(yticklabels,crs=self.map_proj)
        self.ax.tick_params(labelsize=self.labelsize)
        self.ax.set_xlabel('')
        self.ax.set_ylabel('')
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        self.ax.xaxis.set_major_formatter(lon_formatter)
        self.ax.yaxis.set_major_formatter(lat_formatter)
        self.ax.grid( lw=0.5, color='#EBE7E0', alpha=0.5, linestyle='-.')
        ## Take care of the colorbar
        fig = self.ax.figure
        ## rounding the colorbar ticks
        s1 = pd.DataFrame(ranges)
        s2 = s1.applymap(lambda x: rounding(x))[0].tolist()
        cbar_ticks=list(map(str,s2))
        cbar_ticks = [i.rstrip('0').rstrip('.') for i in cbar_ticks]
        if len(cbar_ticks) > 12:
            cbar_ticks[::2]=['']*len(cbar_ticks[::2])
        else:
            cbar_ticks[0]=''
            cbar_ticks[-1]=''
        ## Dynamic page size depending on the lat/lon ranges or panel size
        if self.figsize != None:
            positions = self.ax.get_position()
            gapy = positions.y0-positions.y1
            gapx = positions.x0-positions.x1
            ratio = gapy/gapx
            if (ratio < 0.6) and (ratio > 0.4):
                self.figsize.set_size_inches(18,10,forward=True)
                plt.draw()
            elif (ratio < 0.4):
                self.figsize.set_size_inches(18,7,forward=True)
                plt.draw()
            elif (ratio > 1) and (ratio < 1.3):
                self.figsize.set_size_inches(16,16,forward=True)
                plt.draw()
            elif (ratio > 1.3):
                self.figsize.set_size_inches(14,18,forward=True)
                plt.draw()

        positions = self.ax.get_position()
        if bool(self.colbar) != False:
            cax = fig.add_axes([positions.x0,positions.y0-self.bdist,positions.x1-positions.x0,self.th])
            cbar = fig.colorbar(im,cax=cax,orientation='horizontal',ticks=ranges,extend='neither',fraction=0.08)
            cbar.ax.set_xticklabels(cbar_ticks, size=self.labelsize,rotation=self.trot)
            cbar.set_label(label=self.unit,size=self.labelsize)
            cbar.outline.set_linewidth(1.5)
            cbar.dividers.set_linewidth(1.5)
        ## panel box thickness
        plt.setp(self.ax.spines.values(),lw=1.5)
        return im

        