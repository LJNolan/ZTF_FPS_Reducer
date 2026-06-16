#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:04:25 2026

@author: ljnolan

Processing forced photometry from ZTF
"""
import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from supernolan import get_cbcc


colnames = [
    "sindex", "field", "ccdid", "qid", "filter", "pid", "infobitssci",
    "sciinpseeing", "scibckgnd", "scisigpix",
    "zpmaginpsci", "zpmaginpsciunc", "zpmaginpscirms",
    "clrcoeff", "clrcoeffunc", "ncalmatches",
    "exptime", "adpctdif1", "adpctdif2", "diffmaglim", "zpdiff",
    "programid", "jd", "rfid", "diffimgstatus",
    "forcediffimflux", "forcediffimfluxunc", "forcediffimsnr",
    "forcediffimchisq",
    "forcediffimfluxap", "forcediffimfluxuncap", "forcediffimsnrap",
    "aperturecorr",
    "dnearestrefsrc", "nearestrefmag", "nearestrefmagunc",
    "nearestrefchi", "nearestrefsharp",
    "refjdstart", "refjdend", "procstatus",
]

def txt_to_pd(file, colnames=colnames):
   df = pd.read_csv(file,
                    comment="#",           # ignore header/comments
                    sep='\s+',
                    header=None,
                    names=colnames,
                    na_values=["null"])    # turn 'null' into NaN
   
   # Remove rows with non-numeric `sindex`, primarily the row defining colnames
   df = df[pd.to_numeric(df["sindex"], 
                         errors="coerce").notna()].reset_index(drop=True)
   # Occasionally, files terminate unexpectedly - remove resultant incomplete
   # rows
   df = df.dropna().reset_index(drop=True)
   # Convert all columns except 'filter' to int/float
   exclude = ['filter', 'procstatus']
   for col in colnames:
      if col not in exclude:
         df[col] = pd.to_numeric(df[col])
   df["procstatus"] = (df["procstatus"].astype(str)
                       .str.split(",")
                       .apply(lambda xs: np.array([int(x) for x in xs],
                                                  dtype=int)))
   return df


def group_ra_dec(ra_dec_list, tol=0.1*u.arcsec):
   coords = SkyCoord(ra=[r for r, d in ra_dec_list] * u.deg,
                     dec=[d for r, d in ra_dec_list] * u.deg,
                     frame="icrs")

   n = len(coords)
   used = np.zeros(n, dtype=bool)
   rows = []

   for i in range(n):
      if used[i]:
         continue

      sep = coords[i].separation(coords)
      group = (sep <= tol) & (~used)
      idx = np.where(group)[0]
      used[idx] = True

      rows.append({"ra": float(coords[i].ra.deg),
                   "dec": float(coords[i].dec.deg),
                   "count": len(idx),
                   "indices": idx.tolist()})

   return pd.DataFrame(rows)


def find_missing_pairs(ra_dec_true, ra_dec_list, jds_jde_true,
                       tol=0.1*u.arcsec):
   true_coords = SkyCoord(ra=[r for r, d in ra_dec_true] * u.deg,
                          dec=[d for r, d in ra_dec_true] * u.deg,
                          frame="icrs")

   list_coords = SkyCoord(ra=[r for r, d in ra_dec_list] * u.deg,
                          dec=[d for r, d in ra_dec_list] * u.deg,
                          frame="icrs")

   idx, sep2d, _ = true_coords.match_to_catalog_sky(list_coords)
   matched = sep2d <= tol

   rows = []
   for i, is_match in enumerate(matched):
      if not is_match:
          rows.append({"ra_true": ra_dec_true[i][0],
                       "dec_true": ra_dec_true[i][1],
                       "jdstart": jds_jde_true[i][0],
                       "jdend": jds_jde_true[i][1],
                       "nearest_ra": ra_dec_list[idx[i]][0],
                       "nearest_dec": ra_dec_list[idx[i]][1],
                       "sep_arcsec": sep2d[i].arcsec})

   return pd.DataFrame(rows)


def reorder_peaks(ra_dec_list, ra_dec_peaks, peaks, tol=0.1*u.arcsec):
   list_coords = SkyCoord(ra=[r for r, d in ra_dec_list] * u.deg,
                          dec=[d for r, d in ra_dec_list] * u.deg,
                          frame="icrs")

   peak_coords = SkyCoord(ra=[r for r, d in ra_dec_peaks] * u.deg,
                          dec=[d for r, d in ra_dec_peaks] * u.deg,
                          frame="icrs")

   idx, sep2d, _ = list_coords.match_to_catalog_sky(peak_coords)

   if np.any(sep2d > tol):
       bad = np.where(sep2d > tol)[0]
       raise ValueError(f"{len(bad)} coordinates in ra_dec_list had no match",
                        f"within {tol}")

   reordered_peaks = [peaks[i] for i in idx]
   return reordered_peaks


def ztf_fp_reduction(file, auto_base=True, peak=None, time_frac=0.2,
                     base_start=None, base_end=None, min_epochs=30,
                     flag_thresh=33554432, snspp_thresh=25, medsee_thresh=4,
                     close_less1=0.9, close_more1=1.1, snt=3, snu=5,
                     plot=False, save=True, output_name=None):
   '''
   Performs standard data reduction of ZTF Forced-Photometry data to produce
   calibrated magnitudes. Note that the magnitude error is reported as -1 if
   the magnitude is an upper limit.  Defaults generally follow best practices
   laid out in documentation.
   
   https://irsa.ipac.caltech.edu/data/ZTF/docs/ztf_zfps_userguide.pdf

   Parameters
   ----------
   file : str
      Path to the ZTF FP file to process.
   
   auto_base : bool, optional
      Toggle to automatically approximate the baseline period for the file:
      assuming relevant event(s) have a rapid rise and (relatively) long decay,
      use first (``time_frac``*100)% or ``min_epochs`` epochs of time series,
      whichever is longer, UNLESS peak is within first (``time_frac``*100*1.5)%
      or ``min_epochs``*1.5 epochs, whichever is longer, of time series, then
      use the final amount instead. If True, one must provide ``peak``, and
      optionally ``time_frac``.  If False, one must provide ``base_start`` and
      ``base_end``. The default is True.
   
   peak : float, optional
      Time of peak signal one wishes to study, used only in automatic
      determination of baseline period. This makes sense for non-repeating
      transients whose source is relatively stable (e.g. another transient
      should be unlikely in the overall time range of the data).  Required if
      ``auto_base`` is True. The default is None.
   
   time_frac : float, optional
      Fraction of the overall time range of the data to use for automatic
      baseline approximation (see ``auto_base``). The default is 0.2.
   
   base_start : float, optional
      Start time, in JD, of manually determined baseline. Ignored if
      ``auto_base`` is True. The default is None.
   
   base_end : float, optional
      End time, in JD, of manually determined baseline. Ignored if
      ``auto_base`` is True. The default is None.
   
   min_epochs : int, optional
      Minimum number of epochs to determine baseline. Only used for returning a
      warning if ``auto_base`` is False. The default is 30, as recommended by
      documentation.
   
   flag_thresh : int, optional
      Quality flag threshold - for only highest-quality data, set to 0. The
      default is 33554432, as recommended by documentation.
   
   snspp_thresh : float, optional
      Spatial noise-sigma per pixel threshold, in DN. The default is 25, as
      recommended by documentation.
   
   medsee_thresh : float, optional
      Median seeing threshold, in arcsec. The default is 4, as recommended by
      documentation.
   
   close_less1 : float, optional
      Documentation references certain values should be ~1, so this sets the
      lower threshold for meeting that condition. The default is 0.9.
   
   close_more1 : float, optional
      See ``close_less1``; this is the upper threshold. The default is 1.1.
   
   snt : float, optional
      Signal-to-noise thershold for a detection. The default is 3, as
      recommended by documentation.
   
   snu : float, optional
      Signal-to-noise threshold for determining upper limits for
      non-detections. The default is 5, as recommended by documentation.
   
   plot : bool, optional
      Toggle to make a plot of the resultant calibrated magnitude lightcurve.
      The default is False.
   
   save : bool, optional
      Toggle to save the calibrated dataframe, with magnitudes, to a new file.
      The default is True.
      
   output_name : str, optional
      File path for desired output file if ``save`` is True. The default is
      None, which assumes the input file has the auto-generated name from ZTF
      FPS, and takes the given numeric identifier and appends
      '_calibrated.csv', e.g. 'req0004726981_lc_calibrated.csv'.

   Returns
   -------
   Output calibrated file, if toggled, and plot, if toggled.

   '''
   df = txt_to_pd(file)
   #print(df[df['filter'] == 'ZTF_r'].head())
   
   # S6.1
   df = df[df['infobitssci'] < flag_thresh]
   df = df[df['scisigpix'] <= snspp_thresh]
   df = df[df['sciinpseeing'] <= medsee_thresh]
   
   # Loop through filters
   filters = df['filter'].unique().tolist()
   for a_filter in filters:
      f_df = df.loc[df["filter"] == a_filter].copy()
      
      # Test plot
      #quick_scatter(f_df['jd'].values - f_df['jd'].values[0],
      #              f_df['forcediffimflux'].values)
      
      # S6.2
      # Perform automatic baseline approximation (see ``auto_base`` 
      # description) or use given start/end
      if auto_base:
         if peak is None:
            raise ValueError('Must provide peak if using automatic baseline '+
                             'approximation')
         jds = f_df['jd'].to_list()[0]
         jde = f_df['jd'].to_list()[-1] + f_df['exptime'].to_list()[-1]
         dur = jde - jds
         start_cutoff = jds + (dur * time_frac * 1.5)
         if peak > start_cutoff:
            if len(f_df[f_df['jd'] < (jds + (dur * time_frac))]) > min_epochs:
               baseline = f_df[f_df['jd'] < (jds + (dur * time_frac))]
            else:
               baseline = f_df.iloc[:min_epochs]
         else:
            if len(f_df[f_df['jd'] > (jde - (dur * time_frac))]) > min_epochs:
               baseline = f_df[f_df['jd'] > (jde - (dur * time_frac))]
            else:
               baseline = f_df.iloc[-min_epochs:]
         base_start = baseline['jd'].to_list()[0]
         base_end = baseline['jd'].to_list()[-1]
      
      else:
         if base_start is None or base_end is None:
            raise ValueError('Must provide base_start and base_end if not '+
                             'using automatic baseline approximation')
         baseline = f_df.loc[(f_df["jd"] >= base_start) &
                             (f_df["jd"] <= base_end)]
         if len(baseline) < min_epochs:
            raise RuntimeWarning('Manually determined baseline is shorter '+
                                 'than ``min_epochs``.')
      
      # Determine any offset of baseline from 0
      base_flux = baseline['forcediffimflux'].values
      #quick_hist(base_flux)
      lo, hi = np.percentile(base_flux, [5, 95])
      avg_trimmed_base = base_flux[(base_flux > lo) & (base_flux < hi)].mean()
      
      # Correct offset in data for appropriate filter
      df.loc[df['filter'] == a_filter, 'forcediffimflux'] -= avg_trimmed_base
      
      # S6.3
      # I subdivide these into sub-steps by paragraph, beginning the with the
      # paragraph which starts with "First..."
      
      # S6.3.1
      # This references using an average or some robust equivalent.  For now, I
      # use trimmed average.
      fdix2 = df.loc[df['filter'] == a_filter, 'forcediffimchisq'].values
      lo, hi = np.percentile(fdix2, [5, 95])
      avg_trimmed_fdix2 = fdix2[(fdix2 > lo) & (fdix2 < hi)].mean()
      if not close_less1 < avg_trimmed_fdix2 < close_more1:
         print('Correction required for forcediffimfluxunc, with ' +
               f'<forcediffimchisq> = {avg_trimmed_fdix2}')
         df.loc[df['filter'] == a_filter, 'forcediffimfluxunc'] *= \
                                                   avg_trimmed_fdix2 ** 0.5
      
      # S6.3.2
      # re-generate baseline dataframe
      f_df = df.loc[df["filter"] == a_filter].copy()
      baseline = f_df.loc[(f_df["jd"] >= base_start) &
                          (f_df["jd"] <= base_end)]
      
      # determine RMS of S/N ratios
      sn_ratios = baseline['forcediffimflux'].values / \
                  baseline['forcediffimfluxunc'].values
      p16, p84 = np.percentile(sn_ratios, [16, 84])
      rms = (p84 - p16) / 2
      
      # Apply correction if needed
      if not close_less1 < rms < close_more1:
         print('Correction required for forcediffimfluxunc, with ' +
               f'RMS of S/N ratios = {rms}')
         df.loc[df['filter'] == a_filter, 'forcediffimfluxunc'] *= rms
      
      # S6.3.3
      # =============================
      # I have no idea how to do this
      # =============================
      
   # S6.4
   # Generate calibrated magnitudes and error as columns (not filter-split)
   # Write error as -1 if mag is an upper limit
   
   flux = df["forcediffimflux"].to_numpy()
   fluxunc = df["forcediffimfluxunc"].to_numpy()
   zpdiff = df["zpdiff"].to_numpy()
   
   snr = flux / fluxunc
   det = (snr > snt) & (flux > 0) & (fluxunc > 0)
   
   df["mag"] = np.where(det, zpdiff - 2.5 * np.log10(flux),
                             zpdiff - 2.5 * np.log10(snu * fluxunc))
   df["sigma_mag"] = np.where(det, 1.0857 * fluxunc / flux,
                                   -1.0)
   
   if plot:
      mag = df['mag'].to_numpy()
      sigma = df['sigma_mag'].to_numpy()
      jd = df['jd'].to_numpy()
      start = jd[0]
      jd -= start
      
      cbcc = get_cbcc()
      
      fig, ax = plt.subplots()
      ax.set_xlabel(f'Days since JD {start}')
      ax.set_ylabel('Calibrated magnitude')
      # Distinguish between upper limits and detections
      limits = sigma < 0
      ax.plot(jd[limits], mag[limits], ls="None", marker="v", mfc="none",
              mec=cbcc[1], ms=4)
      ax.errorbar(jd[~limits], mag[~limits], yerr=sigma[~limits], ls="None",
                  marker='o', ms=3, mfc=cbcc[0], capsize=2, mec=cbcc[0],
                  ecolor=cbcc[0])
      plt.show()
      plt.close()
   
   if save:
      if output_name is None:
         output_name = file.replace("batchfp", "").replace(".txt",
                                                           "_calibrated.csv")
      df.to_csv(output_name, index=False)
   print(f'Finished {file}')
   return


def calib_plot(file, peak=None):
   df = pd.read_csv(file)
   mag = df['mag'].to_numpy()
   sigma = df['sigma_mag'].to_numpy()
   jd = df['jd'].to_numpy()
   start = jd[0]
   jd -= start
   
   cbcc = get_cbcc()
   
   fig, ax = plt.subplots()
   ax.set_xlabel(f'Days since JD {start}')
   ax.set_ylabel('Calibrated magnitude')
   # Distinguish between upper limits and detections
   limits = sigma < 0
   if peak is not None:
      peak -= start
      ax.axvline(x=peak, c=cbcc[2], linestyle='--')
   ax.plot(jd[limits], mag[limits], ls="None", marker="v", mfc="none",
           mec=cbcc[1], ms=4)
   ax.errorbar(jd[~limits], mag[~limits], yerr=sigma[~limits], ls="None",
               marker='o', ms=3, mfc=cbcc[0], capsize=2, mec=cbcc[0],
               ecolor=cbcc[0])
   ax.yaxis.set_inverted(True)
   plt.show()
   plt.close()
   return


def calibrate(file):
   df = pd.read_csv(file)
   sigma = df['sigma_mag'].to_numpy()
   jd = df['jd'].to_numpy()
   # Distinguish between upper limits and detections
   limits = sigma < 0
   frac = 100 * len(jd[~limits]) / len(jd)
   return frac


data_dir = 'data' # Directory with all data files
data_dir = Path(data_dir)
files = sorted(data_dir.glob("*.txt"))  # retrieve all .txt files in data_dir

# Get RA/Dec and JDs/JDe information for validation
ra_dec_list = []
jds_jde_list = []

for f in files:
   ra = dec = jds = jde = None
   with f.open() as fh:
      for line in fh:
         line = line.strip()
         if line.startswith("# Requested input R.A."):
            # RegEx to extract RA value
            m = re.search(r"=\s*([0-9.+-Ee]+)", line)
            if m:
               ra = float(m.group(1))
         elif line.startswith("# Requested input Dec."):
            # As above for DEC
            m = re.search(r"=\s*([0-9.+-Ee]+)", line)
            if m:
               dec = float(m.group(1))
         elif line.startswith("# Requested JD start"):
            # As above for JD start
            m = re.search(r"=\s*([0-9.+-Ee]+)", line)
            if m:
               jds = float(m.group(1))
         elif line.startswith("# Requested JD end"):
            # As above for JD end
            m = re.search(r"=\s*([0-9.+-Ee]+)", line)
            if m:
               jde = float(m.group(1))
         if ra is not None and dec is not None:
            break
   if ra is None or dec is None:
      raise ValueError(f"Could not find RA/Dec in {f}")
   ra_dec_list.append((ra, dec))
   jds_jde_list.append((jds, jde))


# Useful later for S6.2...
# Pull peaks and sort
peak_tab = pd.read_csv('peaks.txt', names=['RA', 'DEC', 'peak'], sep=' ')
peak_tab.sort_values('peak', inplace=True) # ensure sorting by peak time
ra_dec_peaks = list(zip(peak_tab['RA'].to_list(), peak_tab['DEC'].to_list()))
peaks = peak_tab['peak'].to_list()
peaks = reorder_peaks(ra_dec_list, ra_dec_peaks, peaks)

# Loop through files
files = [str(f) for f in files]

for n, file in enumerate(files):
   output_name = file.replace("batchfp_", "reduced/").replace(".txt",
                                                     "_calibrated.csv")
   ztf_fp_reduction(file, peak=peaks[n], output_name=output_name)

