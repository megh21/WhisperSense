#%%
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from scipy import signal
import soundfile as sf
import wave
import librosa
import matplotlib.pyplot as plt
import ipywidgets as widgets
import struct
import random
import os
from matplotlib.patches import Rectangle

folder_path = os.getcwd()  # Replace with the actual folder path
Fs=22500
# Get a list of files in the folder
files = os.listdir(folder_path)

# Filter the list to include only WAV files with "cricket" in their names
filtered_files = [file for file in files if file.lower().endswith(".wav") and "mollcricket" in file.lower()]


# # 'mollcricket_45a90.wav'
# filename= random.choice(filtered_files)
# print("The selected file is : ",filename)#,"with distance and angle as ",filename.split("_"[1].split(".")[0]))

# y_stereo,sr  = librosa.load(filename, mono=False)
# Fs = sr
# processed=y_stereo.T[500:]


###################

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut=1000, highcut=9000, fs=Fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def plot_signal (signal, microphones, xlim):
    plt.figure(figsize = (10,4))
    for microphoneNumber in microphones:
        microphone_signal = signal[:,microphoneNumber]
        t =  (1/Fs)*np.arange(len(microphone_signal))
        plt.plot(t,microphone_signal, label = "Mic{}".format(microphoneNumber))
    plt.title('Display the waveforms recorded by the microphones ')
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.xlim(xlim)

def plot_fft(signal, microphone, plot_absFFT_only = False): #BLK
    f = np.arange(0, Fs, Fs/len(signal))
    # Compute fft
    s = signal[:,microphone]
    fft = np.fft.fft(s)
    print("shape of fft here is ",fft.shape )
    if plot_absFFT_only == True : nbGraphe = 1
    else: nbGraphe = 2
            
    # plot the absolute value of fft and its spectrum
    #plt.figure(figsize = (10,4))
    plt.subplot(1,nbGraphe,1)
    plt.plot(f, np.abs(fft), label = "M{}".format(microphone))
    plt.title("Amplitude du spectre du signal M{}".format(microphone))
    plt.xlabel("Frequency(Hz)")
    plt.ylabel("$|TFD(M{})|$".format(microphone))
    plt.xlim(0, 1500)
    plt.legend()
    plt.grid(True)  

    if  plot_absFFT_only == False:
        plt.subplot(1,nbGraphe,2)
        plt.phase_spectrum(s, Fs=Fs, color='C1')
        plt.title("Phase du Spectre du signal M{}".format(microphone))
        plt.xlabel("Frequency(Hz)")
        plt.grid(True)


def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, mono=False)
    # audio=butter_bandpass_filter(audio, lowcut=1000, highcut=9000, fs=Fs, order=5)
    # if len(audio.shape) > 1:
    #     audio = librosa.to_mono(audio)
    stft = np.abs(librosa.stft(audio))
    spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
    return audio, spectrogram, sr

def estimate_distance(stereo_audio, sr):
    intensity_left = np.mean(stereo_audio[0]**2)
    intensity_right = np.mean(stereo_audio[1]**2)
    average_intensity = (intensity_left + intensity_right) / 2.0
    reference_distance = 5.6#245.0
    norm_fact=40
    distance = reference_distance * 1 / np.sqrt(average_intensity*norm_fact)
    return distance



def crossco(wav):
    """Returns cross correlation function of the left and right audio. It
    uses a convolution of left with the right reversed which is the
    equivalent of a cross-correlation.
    """
    cor = np.abs(signal.fftconvolve(wav[0],wav[1][::-1]))
    return cor


def trackTD(fname, width, chunksize=5000):
    track = []
    #opens the wave file using pythons built-in wave library
    wav = wave.open(fname, 'r')
    #get the info from the file, this is kind of ugly and non-PEPish
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams ()

    #only loop while you have enough whole chunks left in the wave
    while wav.tell() < int(nframes/nchannels)-chunksize:

        #read the audio frames as asequence of bytes
        frames = wav.readframes(int(chunksize)*nchannels)

        #construct a list out of that sequence
        out = struct.unpack_from("%dh" % (chunksize * nchannels), frames)

        # Convert 2 channels to numpy arrays
        if nchannels == 2:
            #the left channel is the 0th and even numbered elements
            left = np.array (list (out[0::2]))
            #the right is all the odd elements
            right = np.array (list  (out[1::2]))
        else:
            left = np.array (out)
            right = left

        #zero pad each channel with zeroes as long as the source
        left = np.concatenate((left,[0]*chunksize))
        right = np.concatenate((right,[0]*chunksize))

        chunk = (left, right)

        #if the volume is very low (800 or less), assume 0 degrees
        if np.abs(np.max(left)) < 800 :
            a = 0.0
        else:
            #otherwise computing how many frames delay there are in this chunk
            cor = np.argmax(crossco(chunk)) - chunksize*2
            #calculate the time
            t = cor/framerate
            #get the distance assuming v = 340m/s sina=(t*v)/width
            sina = t*340/width
            sina = -1 if sina <= -1 else 1 if sina >= 1 else sina
            a = -np.arcsin(sina) * 180/(3.14159)
        #add the last angle delay value to a list
        track.append(a)
    #plot the list
    # plot(track)
    return track

#####################


final_array=[]
for filename in filtered_files:

    # filename= random.choice(filtered_files)
    print("The selected file is : ",filename)#,"with distance and angle as ",filename.split("_"[1].split(".")[0]))

    y_stereo,sr  = librosa.load(filename, mono=False)
    Fs = sr
    processed=y_stereo.T[500:]
        
    track=trackTD(filename,width=0.04)
    ######################




    ###############
    plt.figure()
    # plt.plot(t, estimatedAngle_etu, label = 'Our implementation')
    # plt.plot(t, estimatedAngle_prof, label = 'Beamformer')
    plt.plot(track,label = 'cross_correlation')
    plt.xlabel("time(s)")
    plt.ylabel("Angle")
    plt.title("Angle estimation with beamformer")   
    plt.legend()
    plt.show()

    ##########
    from collections import Counter
    b = Counter(track)
    absmax=max(np.abs(track))
    final_ang = np.mean(track)
    print("final_ang is : ",final_ang)

    #############

    audio, spectrogram, sr = preprocess_audio(filename)
    distance = estimate_distance(audio, sr)
    print("distance : ",distance)

    #############

    final_array.append([distance,final_ang])
#%%
# Create the figure and axes
fig, ax = plt.subplots(figsize=(4.5, 4.5))

# Set the aspect ratio of the plot to 'equal' for a square shape
ax.set_aspect('equal')

# Define the grid parameters
rows = 2
columns = 4

# Calculate the width and height of each rectangle
rect_width = 4.5 
rect_height = 4.5 

# Create the rectangles
for i in range(rows):
    for j in range(-2,2):
        rect = Rectangle((j * rect_width, i * rect_height), rect_width, rect_height, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

# Set the limits of the plot
ax.set_xlim(-9, 9)
ax.set_ylim(0, 4.5*rows)

# Draw a line on the rectangle
line_x = [0, 5]  # x-coordinates of the line endpoints
line_y = [0, 5]  # y-coordinates of the line endpoints
ax.plot(line_x, line_y, color='red', linewidth=2)

# # Set the labels for the axes
# ax.set_xlabel('Width')
# ax.set_ylabel('Height')

def line_on_rect(angle_deg,length):
    # Define the starting point and angle
    start_point = (0, 0)  # (x, y) coordinates of the starting point
    # angle_deg = 45  # Angle in degrees

    # Calculate the endpoint of the line
    # length = 5  # Length of the line
    angle_rad = np.radians(angle_deg)  # Convert angle to radians
    end_point = (start_point[0] + length * np.cos(angle_rad), start_point[1] + length * np.sin(angle_rad))

    # Draw the line
    line_x = [start_point[0], end_point[0]]  # x-coordinates of the line endpoints
    line_y = [start_point[1], end_point[1]]  # y-coordinates of the line endpoints
    ax.plot(line_x, line_y, color='red', linewidth=2)
    
for (dist,angle) in final_array:
    if angle >= 0:
        final_angle = 90 - angle
    else:
        final_angle = abs(angle) + 90
    line_on_rect(final_angle, dist*2/10)

# Show the plot
plt.show()

# %%
