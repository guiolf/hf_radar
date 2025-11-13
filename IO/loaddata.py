#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 17:04:01 2025

@author: glopez
"""

#%% Import libraries

import xarray as xr
import numpy as np
import pandas as pd
import time
from pathlib import Path
import datetime

#%% Define class and reading functions

class DataLoader:
    """
    A class to load NetCDF files containing surface current measurements from HFR.
    There are functions to read monthly files previously dowloaded from CMEMS and 
    also gap-filled data, also with harmonized common data model used for the CMEMS
    files.
    
    Designed to handle various time ranges, from a single month
    to multiple years, by dynamically generating the list of files to load
    """
    
    def __init__(self, site, year_start, month_start, year_end, month_end, 
                 base_path='/Users/glopez/Documents/data/totals_cmems/'):
        """
        Initializes the DataLoader with the specified site and date range.
        """
        self.site = site
        self.year_start = int(year_start)
        self.month_start = int(month_start)
        self.year_end = int(year_end)
        self.month_end = int(month_end)
        self.base_path = Path(base_path)
        self.ds = None

    def _get_file_paths(self):
        
        """
        Generates a list of NetCDF file paths based on the specified date range.
        """
        ncfiles = []
        try:
            date_range = pd.date_range(
                start=f'{self.year_start}-{self.month_start}-01',
                end=f'{self.year_end}-{self.month_end}-01',
                freq='MS'
            )
        except ValueError as e:
            print(f"Error: Invalid date range provided. {e}")
            return []

        # Iterate over each month in the generated date range
        
        for dt in date_range:
            
            year_str = f'{dt.year:4d}'
            # Re-enabled this to build the specific file name pattern
            year_month_str = f'{dt.year}{dt.month:02d}' 
            
            # Construct the path to the YEARLY directory
            year_dir = self.base_path / self.site / year_str
            
            # Create a glob pattern to match files for this specific month
            # (e.g., "*202301.nc")
            file_pattern = f"*{year_month_str}.nc"
            
            if year_dir.is_dir():
                # Use the new pattern to find only this month's files
                for ncfile in year_dir.glob(file_pattern):
                    ncfiles.append(str(ncfile))
            else:
                # Updated print message to use the correct variable
                print(f"Warning: Directory not found, skipping: {year_dir}")
                
        return ncfiles
    
    def read_divand(self):
        """
        Reads and processes NetCDF files obtained with DIVAnd into a single Dataset.
        """
        print(f"Starting data load for site '{self.site}' from {self.year_start}-{self.month_start} to {self.year_end}-{self.month_end}.")
        
        ncfiles = self._get_file_paths()
        
        if not ncfiles:
            print("Error: No files found for the specified date range. Aborting.")
            self.ds = None
            return

        print(f"Found {len(ncfiles)} files to load.")

        def fix_time(ds):
            _, index = np.unique(ds['time'], return_index=True)
            return ds.isel(time=index)

        print("Reading and concatenating files...")
        t_start = time.time()
        try:
            self.ds = xr.open_mfdataset(
                ncfiles, autoclose=True, preprocess=fix_time, concat_dim="time",
                combine="nested",  engine='netcdf4'
            )
        except Exception as e:
            print(f"An error occurred while loading files: {e}")
            self.ds = None
            return
        
        elapsed = time.time() - t_start
        print(f"Data concatenation finished in {elapsed:.2f} seconds.")
        
        full_time_range = pd.date_range(start=self.ds['time'].min().values, 
                                        end=self.ds['time'].max().values, freq='1h')
        self.ds = self.ds.reindex(time=full_time_range)
        
        print("Processing complete. Dataset is ready.")
        
        
    def read_divand_cmems(self):
        """
        Reads and processes NetCDF files obtained with DIVAnd into a single Dataset.
        """
        print(f"Starting data load for site '{self.site}' from {self.year_start}-{self.month_start} to {self.year_end}-{self.month_end}.")
        
        ncfiles = self._get_file_paths()
        
        if not ncfiles:
            print("Error: No files found for the specified date range. Aborting.")
            self.ds = None
            return

        print(f"Found {len(ncfiles)} files to load.")

        def fix_time(ds):
            _, index = np.unique(ds['TIME'], return_index=True)
            return ds.isel(TIME=index)
        
        print("Reading and concatenating files...")
        t_start = time.time()
        try:
            self.ds = xr.open_mfdataset(
                ncfiles, autoclose=True, preprocess=fix_time, concat_dim="TIME",
                combine="nested",  engine='netcdf4'
            )
        except Exception as e:
            print(f"An error occurred while loading files: {e}")
            self.ds = None
            return
        
        elapsed = time.time() - t_start
        print(f"Data concatenation finished in {elapsed:.2f} seconds.")
        
        # Standardize and clean the dataset 
        self.ds = self.ds.rename({'LATITUDE': 'lat', 'LONGITUDE': 'lon', 'TIME': 'time',
                                  'EWCT':'U' , 'NSCT': 'V'})
        
        if 'DEPTH' in self.ds.dims:
            self.ds = self.ds.isel(DEPTH=0, drop=True)
        
        full_time_range = pd.date_range(start=self.ds['time'].min().values, 
                                        end=self.ds['time'].max().values, freq='1h')
        self.ds = self.ds.reindex(time=full_time_range)
        
        print("Processing complete. Dataset is ready.")


    def read_cmems(self):
        """
        Reads and processes HFR CMEMS NetCDF files into a single Dataset.
        """
        print(f"Starting data load for site '{self.site}' from {self.year_start}-{self.month_start} to {self.year_end}-{self.month_end}.")
        
        ncfiles = self._get_file_paths()
        
        if not ncfiles:
            print("Error: No files found for the specified date range. Aborting.")
            self.ds = None 
            return

        print(f"Found {len(ncfiles)} files to load.")
        
        vars_dropped = [
            "NARX", "NATX", "SLTR", "SLTT", "SLNR", "SLNT", "SCDR", "SCDT", 
            "SDN_CRUISE", "SDN_XLINK", "SDN_REFERENCES", "SDN_EDMO_CODE", 
            "SDN_LOCAL_CDI_ID", "SDN_STATION", "DEPTH_QC", "TIME_QC", "crs", 
            "GDOP", "DDNS_QC", "CSPD_QC", "VART_QC", "GDOP_QC", "EWCS", 
            "NSCS", "CCOV", "POSITION_QC"
        ]

        def fix_time(ds):
            _, index = np.unique(ds['TIME'], return_index=True)
            return ds.isel(TIME=index)

        print("Reading and concatenating files...")
        t_start = time.time()
        try:
            self.ds = xr.open_mfdataset(
                ncfiles, autoclose=True, preprocess=fix_time, concat_dim="TIME",
                combine="nested", drop_variables=vars_dropped, engine='netcdf4'
            )
        except Exception as e:
            print(f"An error occurred while loading files with xarray: {e}")
            self.ds = None
            return
        
        elapsed = time.time() - t_start
        print(f"Data concatenation finished in {elapsed:.2f} seconds.")

        # Standardize and clean the dataset 
        self.ds = self.ds.rename({'LATITUDE': 'lat', 'LONGITUDE': 'lon', 'TIME': 'time'})
        
        if 'DEPTH' in self.ds.dims:
            self.ds = self.ds.isel(DEPTH=0, drop=True)
        
        # Keep only data flagged as good (QC=1) or probably good (QC=2)
        
        if 'QCflag' in self.ds:
            self.ds['EWCT'] = self.ds['EWCT'].where((self.ds['QCflag'] == 1) | (self.ds['QCflag'] == 2))
            self.ds['NSCT'] = self.ds['NSCT'].where((self.ds['QCflag'] == 1) | (self.ds['QCflag'] == 2))
        
        # Rename variables
        self.ds = self.ds.rename({'EWCT': 'U', 'NSCT': 'V'})
        
        # Some dates might be missing, construct a full dataset filled with NaNs
        full_time_range = pd.date_range(datetime.datetime(int(self.ds['time'].dt.year[0]),
                    int(self.ds['time'].dt.month[0]),1,0),
                    datetime.datetime(int(self.ds['time'].dt.year[-1]),12,31,23),freq='h')
        
        self.ds = self.ds.reindex(time=full_time_range)
        
        print("Processing complete. Dataset is ready.")

    def get_data(self):
        """Returns the loaded and processed xarray.Dataset."""
        return self.ds


if __name__ == '__main__':
    
    SITE_ID = "Ibiza" 
    DATA_PATH = "/Users/glopez/Documents/data/totals_cmems/"

    # Load a single month 
    print("Loading Option 1: Single Month")
    loader_single_month = DataLoader(
        site=SITE_ID,
        year_start=2022, month_start=1,
        year_end=2022, month_end=1,
        base_path=DATA_PATH
    )
    loader_single_month.read_totals()
    
    # Load a range of months in the same year 
    print("\nLoading Option 2: Month Range in One Year")
    loader_month_range = DataLoader(
        site=SITE_ID,
        year_start=2022, month_start=1,
        year_end=2022, month_end=3,
        base_path=DATA_PATH
    )
    loader_month_range.read_totals()
   

    # Load a full year 
    print("\nLoading Option 3: Full Year")
    loader_full_year = DataLoader(
        site=SITE_ID,
        year_start=2023, month_start=1,
        year_end=2023, month_end=12,
        base_path=DATA_PATH
    )
    loader_full_year.read_totals()
   
    # Load a range spanning multiple years 
    print("\nLoading Option 4: Multi-Year Range")
    loader_multi_year = DataLoader(
        site=SITE_ID,
        year_start=2022, month_start=12,
        year_end=2023, month_end=2,
        base_path=DATA_PATH
    )
    loader_multi_year.read_totals()
   