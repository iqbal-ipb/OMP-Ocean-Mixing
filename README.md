# OMP (Optimum Multiparameter) Ocean Mixing
Menghitung dan menampilkan percampuran masa air dengan metode omptimum multiparameter analysis. Kode ini merupakan pengayaan dari kode yang sudah ada di situs  https://ocefpaf.github.io/python4oceanographers/blog/2014/03/24/watermass/ 

deskripsi file:
1. maluku.csv : contoh file data yang digunakan
2. omp_ipb.py : file class untuk perhitungan omp
3. test.py    : contoh penggunaan class yang telah dibuat

![](https://github.com/iqbalipb/OMP-Ocean-Mixing/blob/master/omp_maluku.png)

# Installations:
sebelum menggunakan class ini install terlebih dahulu:
1. python 3
2. install anaconda
3. install beberapa modul berikut: pandas, numpy, matplotlib, gsw, brewer2mpl, mpl_toolkits, scipy
4. Download repo ini, kemudian jalankan test.py (anda dapat menggunakan spyder)

  # buka_csv(filenya, grid_kedalaman, jumlah_stasiun)
 
     filenya: masukan alamat file CSV anda
     grid_kedalaman : jumlah data dari kedalaman
     jumlah_stasiun : jumlah longitude (titik CTD)
     
     catatan: pastikan jumlah data keseluruhan = grid_kedalaman * jumlah_stasiun
 
   # mixing(T, S, inds)
   
     T : nilai temperature
     S : Nilai salinitas
     ind : nilai batas masa air
     
   # gambar_utama(self, S, T, depth, latitude, masa_air1, masa_air2, masa_air3)
   
        T : nilai temperature
        S : Nilai salinitas
        depth : data kedalaman
        latitude : data latitude
        masa_air1 : nilai masa air pertama
        masa_air2 : nilai masa air kedua
        masa_air3 : nilai masa air ketiga
       
   # gambar_colorbar(fig, letak, label)
   
        fig : hasil output dari gambar_utama
        letak : 1 -> pojok kiri atas
                2 -> pojok kiri bawah
                3 -> pojok kanan bawah
                4 -> pojok kanan atas
        label  : nama-nama dari masa air
 
