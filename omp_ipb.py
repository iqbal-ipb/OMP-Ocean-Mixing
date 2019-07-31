# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 07:00:07 2019

@author: Muhammad Iqbal, IPB University
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import gsw
import brewer2mpl

from matplotlib.ticker import MaxNLocator
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import interpolate
from netCDF4 import Dataset
import warnings
warnings.filterwarnings('ignore')


class omp_ipb:
    def buka_netcdf(self, file, var_pres, var_temp, var_sal, var_lat, var_lon):
        self.file = file
        fh = Dataset(self.file, mode='r')

        self.depth = fh.variables[var_pres][:]  # extract/copy the data
        self.T = fh.variables[var_temp][:]
        self.S = fh.variables[var_sal][:]
        self.latitude = fh.variables[var_lat][:]
        self.longitude = fh.variables[var_lon][:]

        return self.S, self.T, self.depth, self.latitude, self.longitude

    def buka_csv(self, filenya, grid_kedalaman):
        self.filenya = filenya
        self.grid_kedalaman = grid_kedalaman
        data = pd.read_csv(self.filenya, sep=';')
        temperature = data['Temperature']
        salinity = data['Salinity']
        self.depth = data['Pressure']
        self.depth=self.depth[0:self.grid_kedalaman]
        self.latitude = data['Latitude']
        self.latitude =np.unique(self.latitude)
        self.longitude = data['Longitude']
        self.longitude = np.unique(self.longitude)
        self.jumlah_stasiun = round(len(temperature)/self.grid_kedalaman)
        if(self.jumlah_stasiun*self.grid_kedalaman != len(temperature)):
            import sys
            print("------------------------------------------------------")            
            print("-----------ERROR JUMLAH DATASET-----------------------")
            print("-----------cek jumlah kedalaman dan transek-----------")
            print("------------------------------------------------------")
            sys.exit()
        j=0
        k=0
        data_temp=np.zeros((self.grid_kedalaman,self.jumlah_stasiun))
        data_sal=np.zeros((self.grid_kedalaman,self.jumlah_stasiun))
        for i in range(0,self.grid_kedalaman*self.jumlah_stasiun,self.grid_kedalaman):
            data_temp[0:,k]=temperature[j:i+self.grid_kedalaman]
            data_sal[0:,k]=salinity[j:i+self.grid_kedalaman]
            j=i+self.grid_kedalaman
            k=k+1

        self.S=data_sal
        self.T=data_temp

        return self.S, self.T, self.depth, self.latitude, self.longitude

    def mixing(self, T, S, inds):
        self.T = T
        self.S = S
        self.inds = inds
        self.jumlah_masa_air = inds.shape[1]
        self.l=np.zeros((S.shape[0],S.shape[1],self.jumlah_masa_air))
        if self.jumlah_masa_air==2:
            a = np.r_[self.inds]
            b = np.c_[self.T.ravel(), self.S.ravel()].T
        elif self.jumlah_masa_air==3:
            a = np.r_[self.inds, np.ones((1, self.jumlah_masa_air))]
            b = np.c_[self.T.ravel(), self.S.ravel(), np.ones(self.T.shape).ravel()].T
        else:
            print("------------------------------------------------------")            
            print("-----------ERROR JUMLAH MASA AIR----------------------")
            print("-----------Mohon Masukan 2 atau 3 masa air------------")
            print("------------------------------------------------------")
            import sys
            sys.exit()

        m = np.linalg.solve(a, b)
        for i in range(self.jumlah_masa_air):
             self.l[0:,0:,i] = m[i].reshape(T.shape)
             self.l[0:,0:,i] = ma.masked_outside(ma.masked_invalid(self.l[0:,0:,i]), 0, 1)
             self.l[0:,0:,i] = 100 * self.l[0:,0:,i]

        return self.l

    def gambar_utama(self, S, T, depth, latitude, hasil_mixing):
        s = ma.masked_invalid(self.S).mean(axis=1)
        t = ma.masked_invalid(self.T).mean(axis=1)
        Te = np.linspace(t.min(), t.max(), 10)
        Se = np.linspace(s.min(), s.max(), 10)

        Tg, Sg = np.meshgrid(Te, Se)
        sigma_theta = gsw.sigma0(Sg, Tg)
        cnt = np.linspace(sigma_theta.min(), sigma_theta.max(), 15)

        Reds = brewer2mpl.get_map('Reds', 'Sequential', 9).mpl_colormap
        Blues = brewer2mpl.get_map('Blues', 'Sequential', 9).mpl_colormap
        Greens = brewer2mpl.get_map('Greens', 'Sequential', 9).mpl_colormap

        warna=[Reds, Blues, Greens]

        # Grid for contouring.
        zg, xg = np.meshgrid(self.depth, self.latitude)
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 6), facecolor='w')
        self.ax.invert_yaxis()
        kl=[]
        percent = [50, 60, 70, 80, 90, 100]
        # m = np.zeros(hasil_mixing.shape)
        for i in range(hasil_mixing.shape[2]):
            k = self.ax.contourf(xg, zg, hasil_mixing[0:,0:,i].transpose(), percent, cmap=warna[i], zorder=3)
            k.set_clim(percent[0], percent[-1])
            kl.append(k)
        
        return self.fig, kl, self.ax

    def gambar_colorbar(self, fig, letak, label):
        # Colorbars.
        dy = 0.04
        if letak==0:
            left, bottom, height, width = 0.14, 0.84, 0.13, 0.02
        elif letak==1:
            left, bottom, height, width = 0.14, 0.26, 0.13, 0.02
        elif letak==2:
            left, bottom, height, width = 0.50, 0.26, 0.13, 0.02
        else:
            left, bottom, height, width = 0.50, 0.84, 0.13, 0.02
        kw = dict(orientation='horizontal', extend='min')
        rect=np.zeros((4, len(fig[1])))
        list_cax=[]
        list_cb=[]
        for i in range(len(fig[1])):
            rect[0:,i] = [left, bottom-dy*i, height, width]  # Top.
            caxk = plt.axes(rect[0:,i], facecolor='#E6E6E6')
            list_cax.append(caxk)
            cb = fig[0].colorbar(fig[1][i], cax=list_cax[i], **kw)
            list_cb.append(cb)
            list_cax[i].xaxis.set_ticklabels([])
            new_labels = ['%s%%' % l.get_text() for l in list_cax[i].xaxis.get_ticklabels()]
            list_cb[i].ax.set_ylabel(label[i], rotation=0,labelpad=20,y=1,verticalalignment='center', horizontalalignment='left')
            list_cb[i].ax.yaxis.set_label_position("right")
        list_cax[len(fig[1])-1].xaxis.set_ticklabels(new_labels)

    def gambar_judul(self, ax, depth, latitude, lon):
        self.lon = lon
        dalam = len(depth)*(round(np.ptp(depth)/len(depth)))
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_label_position('left')
        ax.xaxis.set_tick_params(tickdir='out')
        ax.yaxis.set_tick_params(tickdir='out')

        ax.set_xlim(right=latitude[-1])
        ax.set_ylim(bottom=dalam)

        plt.draw()
        new_labels = [r'%s $^\circ$S' % l.get_text()[1:] for l in
                        ax.xaxis.get_ticklabels()]
        ax.xaxis.set_ticklabels(new_labels)

        new_labels = [r'%s m' % l.get_text() for l in
                        ax.yaxis.get_ticklabels()]
        ax.yaxis.set_ticklabels(new_labels)
        ax.set_xlabel(u'Water masses at longitude %3.1f\u00B0' % self.lon)
    def gambar_TS(self, ax, S, T, cores, labels, jumlah_garis_TS, posisi_judul):
        # T-S Diagram.
        s = ma.masked_invalid(S).mean(axis=1)
        t = ma.masked_invalid(T).mean(axis=1)
        Te = np.linspace(t.min(), t.max(), jumlah_garis_TS)
        Se = np.linspace(s.min(), s.max(), jumlah_garis_TS)


        Tg, Sg = np.meshgrid(Te, Se)
        sigma_theta = gsw.sigma0(Sg, Tg)
        cnt = np.linspace(sigma_theta.min(), sigma_theta.max(), jumlah_garis_TS)
        divider = make_axes_locatable(ax)
        axTS = divider.append_axes("right", 2, pad=1)
        #axTS.set(xlim=(33.2,35), ylim=(6,33))
        cs = axTS.contour(Sg, Tg, sigma_theta, colors='grey', levels=cnt, zorder=1)
        kw = dict(color='r', fontsize=14, fontweight='black')
        for i in range(cores.shape[1]):
            axTS.text(cores[1,i], cores[0,i], labels[i], **kw)

        axTS.plot(s, t, 'k')
        axTS.yaxis.set_label_position(posisi_judul)
        axTS.yaxis.set_ticks_position(posisi_judul)
        axTS.set_xlabel("Salinity [g kg$^{-1}$]")
        axTS.set_ylabel("Temperature [$^\circ$C]", rotation=-90, labelpad=20)
        axTS.set_title("T-S Diagram")
        axTS.xaxis.set_major_locator(MaxNLocator(nbins=4))

    def gambar_peta(self, ax, lon, longitude, latitude, mode, lokasi, resolusi):
        #mengatur peta legenda dan lokasi dimana didalam gambar
        axin = inset_axes(ax, width="35%", height="35%", loc=lokasi)
        #-----------------
        xku = Basemap(projection='ortho', lon_0=lon, lat_0=np.median(latitude),
                        ax=axin, anchor='NE', resolution='l')
        if mode==0:
            inmap = Basemap(projection='ortho',lon_0=lon-resolusi,lat_0=np.median(latitude)-resolusi,resolution='l',\
                llcrnrx=0.,llcrnry=0.,urcrnrx=xku.urcrnrx/15.,urcrnry=xku.urcrnry/20.)
            inmap.fillcontinents()
            inmap.plot(longitude, latitude, 'r', latlon=True)
        else:
            xku.fillcontinents()
            xku.plot(longitude, latitude, 'r', latlon=True)
