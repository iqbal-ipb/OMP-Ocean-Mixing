# -*- coding: utf-8 -*-
"""
@author: Muhammad Iqbal, IPB University


"""

import numpy as np
from omp_ipb import omp_ipb

omp_ipb=omp_ipb()
#file bentuk CSV terdiri atas Kedalaman, Temperature, Salnity
S, T, depth, lat, lon = omp_ipb.buka_csv("maluku.csv", 250, 13)
#S, T, depth, lat, lon = omp_ipb.buka_netcdf("data/D6901746_002.nc", 'PRES', 'TEMP', 'PSAL', 'LATITUDE', 'LONGITUDE')
"""
     T : Conservative Temperature
     S : Absolute Salinity
     inds :  2x3 array with thermohaline indices
            [T1 T2 T3
            S1 S2 S3]
"""
#masukan 2 atau 3 masa air
cores = np.array([[20, 8.5, 9], [34.8, 34.5, 34.6]])
labels = ['NPSW', 'NPIW', 'SPSW'] #nama masa air
hasil = omp_ipb.mixing(T, S, cores) #hiung OMP
"""
hasil kemudian ditampilkan dalam bentuk gambar

"""
gbr = omp_ipb.gambar_utama(S,T, depth, lat, hasil)
"""
gambar_colorbar(gbr, 4, labels)

gbr    : hasil dari omp_ipb.gambar_utama
4      : letak color bar (1: kiri atas, 2: kiri bawah, 3 kanan bawah dan 4 kanan atas)
labels : nama nama masa air
"""
omp_ipb.gambar_colorbar(gbr, 4, labels)
omp_ipb.gambar_judul(gbr[2], depth, lat, np.median(lon))
"""
gambar_TS(gbr[2], S, T, cores, labels, 15, "left")

gbr    : hasil dari omp_ipb.gambar_utama
15     : jumlah garis densitas
"left" : letak judul sumbu-Y
"""
omp_ipb.gambar_TS(gbr[2], S, T, cores, labels, 15, "left")

"""
gambar_peta(gbr[2], np.median(lon), lon, lat, 0, 3,2)

gbr    : hasil dari omp_ipb.gambar_utama
0      : untuk 0 --> peta zoom, 1 --> globe
3 	   : letak color bar (1: kiri atas, 2: kiri bawah, 3 kanan bawah dan 4 kanan atas)
2	   : nilai zoom
"""
omp_ipb.gambar_peta(gbr[2], np.median(lon), lon, lat, 0, 3,2)



