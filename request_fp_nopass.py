#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 10:54:33 2026

@author: ljnolan

Based on the ZTF ZFPS User Guide example script to request forced photometry.

Note to self: this version has my password, so I added it to .gitignore, but I
should make a duplicate with a dummy email/password for the GitHub
"""

import requests
import json
import pandas as pd
import datetime as dt
import numpy as np


def submit_post(ra_list, dec_list, jds, jde):
   ra = json.dumps(ra_list)
   dec = json.dumps(dec_list)
   jdstart = json.dumps(jds)
   jdend = json.dumps(jde)
   
   if len(ra) > 1500 or len(dec) > 1500:
      print("Something's wrong with numbers")
      print(f"ra: {len(ra)}\ndec: {len(dec)}")
      return
   
   email = 'liamjn2@illinois.edu' # email you subscribed with.
   userpass = '*******' # password that was issued to you.
   payload = {'ra': ra, 'dec': dec,
   'jdstart': jdstart, 'jdend': jdend,
   'email': email, 'userpass': userpass}
   # fixed IP address/URL where requests are submitted:
   url = 'https://ztfweb.ipac.caltech.edu/cgi-bin/batchfp.py/submit'
   #r = requests.post(url,auth=('ztffps', 'dontgocrazy!'), data=payload)
   #print("Status_code =",r.status_code)
   #print("Response headers:", r.headers)
   #print("Response text:\n", r.text)
   return


def batch_submit(tab, cap):
   # limit to ```cap``` positions per request
   ra_list = tab['RA'].to_list()
   dec_list = tab['DEC'].to_list()
   jds_list = tab['JDS'].to_numpy()
   jde_list = tab['JDE'].to_numpy()
   
   l = len(ra_list)
   jds = min(jds_list)
   while l / cap > 1:
      jde = max(jde_list[:cap+1])
      submit_post(ra_list[:cap+1], dec_list[:cap+1], jds, jde)
      
      ra_list = ra_list[cap+1:]
      dec_list = dec_list[cap+1:]
      jds_list = jds_list[cap+1:]
      jde_list = jde_list[cap+1:]
      
      l -= cap
      jds = min(jds_list)
   jde = max(jde_list)
   submit_post(ra_list, dec_list, jds, jde)
   return


cap = 1500 # limit on number of positions per request
jd_ztf = 2458194.5 # start of ZTF survey
today_timestamp = pd.Timestamp(dt.datetime.utcnow())
jd_today = float(today_timestamp.to_julian_date())
ztf_dur = jd_today - jd_ztf
segments = 2 # how many pieces to split ZTF survey duration into
jd_segments = [jd_ztf + ((ztf_dur * n) / segments) for n in range(segments+1)]

tab = pd.read_csv('targets.txt', names=['RA', 'DEC', 'JDS', 'JDE'], sep=' ')
tab.sort_values('JDS', inplace=True) # ensure sorting by start time
tab_ex = tab.copy()

for n in range(segments):
   print('Starting segment', n+1)
   start = jd_segments[n]
   end = jd_segments[n+1]
   seg_tab = tab[(tab['JDS'] > start) & (tab['JDE'] < end)]
   tab_ex = tab_ex[(tab_ex['JDS'] < start) | (tab_ex['JDE'] > end)]
   
   print(f"Segment {n+1}: {len(seg_tab)} rows")
   batch_submit(seg_tab, cap)

# send leftovers
print(f"Leftovers: {len(tab_ex)} rows")
batch_submit(tab_ex, cap)


#--------------------------------------------------
# Main calling program. Ensure "targets.txt"
# contains your RA Dec positions and desired JD ranges.

# with open('targets.txt') as f:
#    lines = f.readlines()
#    f.close()
# print("Number of targets =", len(lines))
# ralist = []
# declist = []
# jdslist = []
# jdelist = []
# for i, line in enumerate(lines):
#    x = line.split()
   
#    radbl = float(x[0])
#    decdbl = float(x[1])
#    jdsdbl = float(x[2])
#    jdedbl = float(x[3])
   
#    raval = float('%.7f'%(radbl))
#    decval = float('%.7f'%(decdbl))
#    jdsval = float('%.7f'%(jdsdbl))
#    jdeval = float('%.7f'%(jdedbl))
   
#    ralist.append(raval)
#    declist.append(decval)
#    jdslist.append(jdsval)
#    jdelist.append(jdeval)
   
#    rem = (i+1) % 1500 # Limit submission to 1500 sky positions.
#    if rem == 0:
#       submit_post(ralist, declist, jdslist, jdelist)
#       ralist = []
#       declist = []
#       jdslist = []
#       jdelist = []
# if len(ralist) > 0:
#    submit_post(ralist, declist, jdslist, jdelist)
