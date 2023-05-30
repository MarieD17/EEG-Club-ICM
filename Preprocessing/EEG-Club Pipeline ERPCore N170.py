# -*- coding: utf-8 -*-
"""
===============================================================================
Spyder Editor
author = Marie Degrave & Thomas Andrillon (2023)

ICM EEG Club Example pipeline
Data: ERP Core MMN paradigm

===============================================================================
"""

# %% IMPORT MODULES
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
import glob
from autoreject import AutoReject
from mne_icalabel import label_components

#from copy import deepcopy

# %% Paths & Variables
# Paths
if 'thandrillon' in os.getcwd():                                                # flexible loop to find the data on your personal computer
    path_data='/Users/thandrillon/Data/ERPCore/ERPCore_N170/'
elif 'degrave' in os.getcwd():
    path_data='/Users/marie.degrave/Documents/EEG-Club perso/ERPCore_N170/'
else:
    path_data='your_path'                                              

if os.path.exists(path_data+"reports")==False:                                  # create the directory if it doesn't exist 
    os.makedirs(path_data+"reports")
if os.path.exists(path_data+"intermediary")==False:
    os.makedirs(path_data+"intermediary")
    
files = glob.glob(path_data + '*.set')                                          # Return a list of pathnames matching the specified pattern

# %% LOAD, FILTER, CLEAN
report_Event = mne.Report(title='AutEvent')
report_AR = mne.Report(title='Auto Reject')
report_ERP = mne.Report(title='ERP')
report_ICA = mne.Report(title='ICA')

for file in files:

    # [1] LOAD RAW DATA
    file_name=file.split('/')[-1]                                               # Splits the file on / and return the last item (ex : 29_MMN.set)
    sub_ID=file_name.split('_')[0]                                              # Splits the file on _ and return the first item (ex : 29)
    report_prefix=path_data+"reports/"+sub_ID+"_"                               # Create a path
    raw = mne.io.read_raw_eeglab(file, preload=True)                            # Read the EEGLAB .set file
    raw_eeg = raw.copy().drop_channels(['HEOG_left','HEOG_right','VEOG_lower']) # New raw file removing some channels
    
    
    print(raw_eeg)
    #HCGSN256_montage = mne.channels.make_standard_montage('GSN-HydroCel-256')
    #raw.set_montage(HCGSN256_montage)

    # [2] MONTAGE - get the right locaiton of your data
    montage = mne.channels.make_standard_montage('standard_1020')               # Read a built-in standard montage that ships with MNE
    raw_eeg.rename_channels(dict(FP1 = 'Fp1', FP2 = 'Fp2'))                     # Rename channels to match with the montage 
    raw_eeg.set_montage(montage, on_missing='ignore')                           # Add it to the raw object, ignore when channels have missing coordinates

    
    # [3] REREFERNCING AND FILTERING
    raw_eeg.resample(256)                                                       # Resample all channels to speed up computations
    sfreq = raw_eeg.info["sfreq"]                                               # Get the sampling frequency used while recording
    raw_eeg.set_eeg_reference("average")                                        # Set the reference for EEG
    raw_eeg.filter(0.1, 100, fir_design='firwin')                               # Quick filtering with a linear filter (FIR)
    raw_eeg.notch_filter(60,filter_length='auto', phase='zero')                 # Remove electrical artifact
    report = mne.Report(title=sub_ID)                                           # Create one report by subject
    report.add_raw(raw=raw_eeg, title='Filtered Cont from"raw_eeg"', psd=False) # Add raw object to the report, omit PSD plot

    # [4] EVENTS
    stim_events=[range(1,40),range(41,80),range(101,140),range(141,180)]        # Recreate the events values : face, car, scrambled face, scrambled car

    (events, event_dict) = mne.events_from_annotations(raw_eeg)                 # Get events from annotations, and rename them
    events=mne.merge_events(events, list(range(1,41)), 1001, replace_events=True)
    events=mne.merge_events(events, list(range(41,81)), 1002, replace_events=True)
    events=mne.merge_events(events, list(range(101,141)), 1003, replace_events=True)
    events=mne.merge_events(events, list(range(141,181)), 1004, replace_events=True)
    
    count_events=mne.count_events(events)                                       # Count the different types of events
    face_id=1001;
    car_id=1002;
    stim_events = mne.pick_events(events, include=[face_id,car_id])             # Select the events of interest

    report.add_events(events=stim_events, 
                      title='Events from "stim_events"', sfreq=sfreq)           # Add events to the HTML report

    report_Event.add_events(events=stim_events, 
                            title='Events: '+sub_ID, sfreq=sfreq)

    # [5] EPOCHS
    epochs = mne.Epochs(raw_eeg, events, event_id=[face_id,car_id],
                        tmin=-0.2, tmax=1.2, reject=None, preload=True)         # Extract epochs from raw, relative to time-locked events
    report.add_epochs(epochs=epochs, title='Epochs from "epochs"')              # Create a report with the epochs data 
    savename = "e_" + sub_ID + ".fif"
    epochs.save(path_data+"intermediary/"+savename, overwrite=True)             # Save the report
    
    # [6] AUTOREJECT  -  automatically reject bad trials and repair bad channels
    n_interpolates = np.array([1, 4, 32])                                       # Values to try for the number of channels for which to interpolate
    consensus_percs = np.linspace(0, 1.0, 11)                                   # Values to try for perc of channels that must agree 
    ar = AutoReject(n_interpolates, consensus_percs,
                    thresh_method='random_search', random_state=42)
    ar.fit(epochs)                                                              # Fit the epochs on the autoreject object
    epochs_clean, reject_log = ar.transform(epochs, return_log=True)            # Remove bad epochs, repairs sensors and returns clean epochs
    epochs_clean.set_eeg_reference("average")

    savename = "ce_" + sub_ID + ".fif"
    epochs_clean.save(path_data+"intermediary/"+savename, overwrite=True)       # Save the cleaned data
    
    fig = reject_log.plot(orientation = 'horizontal', show=False)               # Plot a resume of good, interpolate and bad channels and epochs

    report.add_figure(                                                          # Add the figure to the report
        fig=fig,
        title="Reject log",
        caption="The rejct log returned by autoreject",
        image_format="PNG",
    )
    report_AR.add_figure(
        fig=fig,
        title="Reject log: "+sub_ID,
        caption="The rejct log returned by autoreject",
        image_format="PNG",
    )
    
    # [8] ICA
    ica_epochs = mne.Epochs(raw_eeg.copy().filter(l_freq=1.0, h_freq=None), 
                            events, event_id=[face_id,car_id],tmin=-0.2, 
                            tmax=1.2, reject=None, preload=True,baseline=None
                            )                                                   # Extracts the epochs
    ica_epochs_clean = ar.transform(ica_epochs)                                 # Remove bad epochs, repairs sensors and returns clean epochs
    ica = mne.preprocessing.ICA(n_components=15, max_iter="auto", 
                                random_state=97,method='infomax', 
                                fit_params=dict(extended=True)
                                )                                               # Set the Independant Component Analysis parameters
    ica.fit(ica_epochs_clean)                                                   # Run the ICA decomposition on raw data
    savename = "ica_ce_" + sub_ID + ".fif"
    ica.save(path_data+"intermediary/"+savename, overwrite=True)                # Save ICA solution 
    
    # ICA rejection
    ica_classification=label_components(ica_epochs_clean, ica, 
                                        method='iclabel'
                                        )                                       # Create a dictionary with the probability of the classes for each independant component
    ica_labels=pd.DataFrame(ica_classification)
    ica_labels.to_csv(report_prefix+'ICAlabels.csv')                            # Add the data structure to the report
    labels = ica_labels["labels"]
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
    caption_str=''
    for idx, label in enumerate(labels):
        caption_str=caption_str+'ICA'+str(idx)+': '+label+'; '
        
    print(f"Excluding these ICA components: {exclude_idx}")
    epochs_clean_ica=ica.apply(epochs_clean, exclude=exclude_idx)
    epochs_clean_ica.set_eeg_reference("average")

    ica_fig=ica.plot_sources(ica_epochs_clean, show_scrollbars=True, 
                             show=False
                             )                                                  # Plot component latent sources over epochs
    report.add_ica(
        ica=ica,
        title='ICA cleaning',
        inst=ica_epochs_clean,
        n_jobs=2  # could be increased!
    )
    report.add_figure(
        ica_fig,
        title="ICA sources",
        caption=caption_str,
    )
    
    report_ICA.add_ica(
        ica=ica,
        title='ICA:' + sub_ID,
        inst=ica_epochs_clean,
        n_jobs=2  # could be increased!
    )
    report_ICA.add_figure(
        ica_fig,
        title='ICA sources:' + sub_ID,
        caption=caption_str,
    )

    # [7] ERP
    evoked_face  = epochs[str(face_id)].average()                               # Compute an average over epochs
    evoked_car = epochs[str(car_id)].average()
    
    evoked_clean_face  = epochs_clean[str(face_id)].average()
    evoked_clean_car = epochs_clean[str(car_id)].average()
    
    evoked_ica_clean_face  = epochs_clean_ica[str(face_id)].average()
    evoked_ica_clean_car = epochs_clean_ica[str(car_id)].average()
    
    conditions=[str(face_id),str(car_id)];
    evoked_clean_perCond = {c:epochs_clean_ica[c].average() for c in conditions}
    savename = "erp_ce_" + sub_ID + ".fif"
    mne.write_evokeds(path_data+"intermediary/"+savename, 
                      list(evoked_clean_perCond.values()), overwrite=True
                     )                                                          # Write the evoked dataset to a file
    report.add_evokeds(
        evokeds=[evoked_car,evoked_face,evoked_clean_car,evoked_clean_face,evoked_ica_clean_car,evoked_ica_clean_face],
        titles=["car", "face","clean car", "clean face",
                "ica+clean car", "ica+clean face"],                             # Manually specify titles
        n_time_points=5,
        replace=True)

    
    # [9] CONTRAST
    picks = 'PO8'
    evokeds_ica_clean = dict(
        car=evoked_ica_clean_face, face=evoked_ica_clean_car
        )                                                                       # Create a dictionary for the standard and deviant signal (MMN)
    erp_ica_clean_fig=mne.viz.plot_compare_evokeds(
        evokeds_ica_clean, picks=picks, show=False
        )                                                                       # Plot the evoked time courses for the two conditions
    
    evokeds = dict(car=evoked_face, face=evoked_car)
    erp_fig=mne.viz.plot_compare_evokeds(evokeds, picks=picks, show=False)

    report.add_figure(
         erp_fig,
         title="ERP contrast",
         caption="Face vs Car at PO8",
     )
    report.add_figure(
          erp_ica_clean_fig,
          title="ERP contrast cleaned+ica",
          caption="Face vs Car at PO8",
      )
 
    face_ica_clean_vis = mne.combine_evoked(
        [evoked_ica_clean_face, evoked_ica_clean_car], weights=[1, -1]
        )                                                                       # Merge evoked data
    erp_ica_clean_but_fig=face_ica_clean_vis.plot_joint(show=False)             # Plot evoked data as butterfly plot and add topomaps for time points
    face_clean_vis = mne.combine_evoked(
        [evoked_clean_face, evoked_clean_car], weights=[1, -1]
        )
    erp_clean_but_fig=face_clean_vis.plot_joint(show=False)
    face_vis = mne.combine_evoked([evoked_face, evoked_car], weights=[1, -1])
    erp_but_fig=face_vis.plot_joint(show=False)
    report.add_figure(
          erp_but_fig,
          title="ERP contrast (butterfly)",
          caption="Face vs Car across all Elec",
      )
    report.add_figure(
          erp_clean_but_fig,
          title="clean ERP contrast (butterfly)",
          caption="Face vs Car across all Elec",
      )
    report.add_figure(
          erp_ica_clean_but_fig,
          title="ica+clean ERP contrast (butterfly)",
          caption="Face vs Car across all Elec",
      )
    savename = "cont_ce_" + sub_ID + ".fif"
    face_ica_clean_vis.save(path_data+"intermediary/"+savename, overwrite=True)
    

    report_ERP.add_figure(
          erp_but_fig,
          title="diff: "+sub_ID,
          caption="Face vs Car across all Elec",
      )
    report_ERP.add_figure(
          erp_clean_but_fig,
          title="cleaned diff: "+sub_ID,
          caption="Face vs Car across all Elec",
      )
    report_ERP.add_figure(
          erp_ica_clean_but_fig,
          title="ica+cleaned diff: "+sub_ID,
          caption="Face vs Car across all Elec",
      )
    
    report.save(report_prefix+"pipeline.html", overwrite=True, open_browser=False)
    
    report_Event.save(path_data+"reports/"+"Events.html", overwrite=True, open_browser=False)
    report_AR.save(path_data+"reports/"+"AutoRej.html", overwrite=True, open_browser=False)
    report_ERP.save(path_data+"reports/"+"ERP.html", overwrite=True, open_browser=False)
    report_ICA.save(path_data+"reports/"+"ICA.html", overwrite=True, open_browser=False)
    
    plt.close('all')
    
# %% GET ERPs across subjects
evokeds_files = glob.glob(path_data+"intermediary/" + '/erp_ce_*.fif')
evokeds = {}                                                                    # create an empty dict
conditions = ['1001','1002']

for idx, c in enumerate(conditions):
    evokeds[c] = [mne.read_evokeds(d)[idx] for d in 
    evokeds_files]                                                              # convert list of evoked in a dict (w/ diff conditions if needed)

evokeds                                                                         # We can see that he matched the conditions by treating each as if it was 2 objcts as before 


ERP_mean = mne.viz.plot_compare_evokeds(evokeds,
                             picks='PO8', show_sensors='upper right',
                             title='Averaged ERP all subjects',
                            )                                                   # Plot averaged ERP on all subj
plt.show()


#gfp: "Plot averaged ERP on all subj"
ERP_gfp = mne.viz.plot_compare_evokeds(evokeds,
                                       combine='gfp', show_sensors='upper right',
                                       title='Averaged ERP all subjects',
                                       )
plt.show()


# evokeds_files = glob.glob(path_data+"intermediary/" + '/cont_ce_*.fif')
# evokeds_diff = {} #create an empty dict
# # #convert list of evoked in a dict (w/ diff conditions if needed)
# for idx, d in enumerate(evokeds_files):
#     evokeds_diff[idx] = mne.read_evokeds(d)[0]
    
# ERP_mean = mne.viz.plot_evoked(evokeds_diff,
#                              picks='PO8',
#                              title='Averaged difference wave all subjects',
#                             )
# plt.show()