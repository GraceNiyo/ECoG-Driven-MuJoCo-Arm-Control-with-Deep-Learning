import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
import mne

def explore_and_plot_bci_data(
    data_folder_path: str,
    subject_number: int,
    sfreq: int = 1000, # Default to 1000 Hz as per dataset
    n_channels_to_plot: int = 5,
    duration_to_plot: int = 10,
    ecog_scaling: float = 10000, # initial assumption for scaling to be validated
    finger_to_plot_index: int = 0 , # 0:Thumb, 1:Index, 2:Middle, 3:Ring, 4:Little
    finger_plot_start: int = 0, # Start index for finger plot
    finger_plot_duration: int = 5
):
    """
    Loads, explores, and visualizes ECoG and Dataglove data from BCI Competition IV Dataset 4.
    """

    #  File Path and Load Data ---
    data_file_name = f'sub{subject_number}_comp.mat'
    data_file_path = os.path.join(data_folder_path, data_file_name)

    try:
        mat_data = scipy.io.loadmat(data_file_path)
        print(f"Data loaded successfully from {data_file_path}")
    except FileNotFoundError:
        print(f"Error: {data_file_path} not found. Please make sure the .mat file is in the correct directory.")
        print(f"Expected path: '{data_folder_path}/sub{subject_number}_comp.mat'")
        print("If you haven't downloaded the data, please refer to the BCI Competition IV website.")
        return 

    # Data Exploration (from .mat file) ---
    print("\n--- Data Structure Overview (from .mat file) ---")
    for key, value in mat_data.items():
        if not key.startswith('__'):
            print(f"Key: {key}, Type: {type(value)}, Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")


    train_ecog = mat_data['train_data']
    dg_train = mat_data['train_dg']
    test_ecog = mat_data['test_data']

    #  Understand Basic Parameters ---
    n_channels_ecog = train_ecog.shape[1]
    n_fingers = dg_train.shape[1]
    finger_labels = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']

    print(f"\n--- Data Parameters ---")
    print(f"Sampling Frequency (sfreq): {sfreq} Hz")
    print(f"Number of ECoG Channels (Subject {subject_number}): {n_channels_ecog}")
    print(f"Number of Fingers (Dataglove): {n_fingers}")
    print(f"Train ECoG duration: {train_ecog.shape[0] / sfreq / 60:.2f} minutes")
    print(f"Test ECoG duration: {test_ecog.shape[0] / sfreq / 60:.2f} minutes")


    ch_names = [f'ECoG {i+1}' for i in range(n_channels_ecog)]
    ch_types = ['ecog'] * n_channels_ecog

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw_ecog = mne.io.RawArray(train_ecog.T, info) # MNE expects (channels, samples)

    # print("\n--- MNE Raw Object Info ---")
    # print(raw_ecog)
    # print(raw_ecog.info)

    # --- Visualize Raw ECoG Data ---
    print("\n--- Visualizing Raw ECoG Data ---")

    # Plot raw ECoG signals
    channels_to_pick = min(n_channels_to_plot, n_channels_ecog)
    raw_ecog.plot(duration=duration_to_plot, n_channels=channels_to_pick,
                 scalings={'ecog': ecog_scaling}, show_scrollbars=True, show_options=True,
                 picks=range(channels_to_pick))
    plt.suptitle(f'Raw ECoG Signals (Subject {subject_number}, First {duration_to_plot}s, First {channels_to_pick} Channels)')
    plt.show()

    # Plot ECoG Power Spectral Density (PSD)
    spectrum = raw_ecog.compute_psd(method='welch', fmax=200, average='mean', picks='ecog')
    spectrum.plot(spatial_colors=False, picks='ecog')
    plt.suptitle(f'ECoG Power Spectral Density (PSD) - Subject {subject_number}')
    plt.show()

    # Explore Dataglove Data (Kinematics) ---
    print("\n--- Exploring Dataglove Data (Kinematics) ---")

    time_dg = np.arange(dg_train.shape[0]) / sfreq

    plt.figure(figsize=(12, 6))
    for i in range(n_fingers):
        plt.plot(time_dg, dg_train[:, i], label=f'{finger_labels[i]} Flexion')
    plt.title(f'Dataglove Finger Flexion - All Fingers (Subject {subject_number})')
    plt.xlabel('Time (s)')
    plt.ylabel('Flexion (Arb. Units)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot a specific finger's flexion for the entire duration
    if 0 <= finger_to_plot_index < n_fingers:
        start_sample_window = int(finger_plot_start * sfreq)
        end_sample_window = min(start_sample_window + int(finger_plot_duration * sfreq), dg_train.shape[0])

        if start_sample_window >= dg_train.shape[0]:
            print(f"Warning: finger_plot_start ({finger_plot_start}s) is beyond the data duration. No plot generated.")
        else:
            plt.figure(figsize=(12, 6))
            plt.plot(time_dg[start_sample_window:end_sample_window],
                     dg_train[start_sample_window:end_sample_window, finger_to_plot_index],
                     label=f'{finger_labels[finger_to_plot_index]} Flexion')
            plt.title(f'{finger_labels[finger_to_plot_index]} Flexion (Subject {subject_number}) from {finger_plot_start}s to {finger_plot_start + finger_plot_duration}s')
            plt.xlabel('Time (s)')
            plt.ylabel('Flexion (Arb. Units)')
            plt.grid(True)
            plt.legend()
            plt.show()
    else:
        print(f"Warning: Invalid finger_to_plot_index ({finger_to_plot_index}). Must be between 0 and {n_fingers-1}.")

