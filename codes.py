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

#%%

filename='mollcricket_45a90.wav'

with wave.open(filename, 'rb') as f:
    frames = f.readframes(-1)
    audio = np.frombuffer(frames, dtype='int16')

y_stereo,sr  = librosa.load(filename, mono=False)
Fs = sr
processed=y_stereo.T[500:]

#%%
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



# %%
plot_signal(processed,microphones = [0,1], xlim=(0,5))
plt.figure(figsize = (10,4))
plot_fft(processed, microphone = 1)

# %%
## Distance ##

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
    reference_distance = 55*5.6#245.0
    extra_factor=60
    distance = reference_distance * np.sqrt(extra_factor*average_intensity)
    return distance

def crossco(wav):
    """Returns cross correlation function of the left and right audio. It
    uses a convolution of left with the right reversed which is the
    equivalent of a cross-correlation.
    """
    cor = np.abs(signal.fftconvolve(wav[0],wav[1][::-1]))
    return cor


def estimate_azimuth_angle(stereo_audio, sr):
    
    cross_correlation = np.correlate(stereo_audio[0], stereo_audio[1], mode='full') #crossco(stereo_audio.T)
    peak_index = np.argmax(cross_correlation)
    delay = (peak_index - len(stereo_audio[0]) + 1) / sr
    speed_of_sound = 343.2
    distance = speed_of_sound * delay
    radius = 0.08
    azimuth_angle = np.arccos(distance / radius)
    azimuth_degrees = np.degrees(azimuth_angle)
    return azimuth_degrees

# %%
audio, spectrogram, sr = preprocess_audio(filename)
distance = estimate_distance(audio, sr)
print("distance : ",distance)
# %%
azimuth_degrees = estimate_azimuth_angle(audio, sr)
azimuth_degrees

# %%
#line:1
N =2 #line:4
d =0.04 #line:5
def beam_filter(OOO0000000O00O0O0 ,OO00000OO0O0OOO00 ,O00O0OO00O0000OOO ,OOOO00OOO0O000OO0 =0 ,OO00O0OO0O0O00O0O :int =0 ):#line:7
    ""#line:19
    OO0OOOO00OOO0O0OO =(OO00O0OO0O0O00O0O -OO00000OO0O0OOO00 -1 )/2 *O00O0OO00O0000OOO #line:22
    return np .exp (-1j *2 *np .pi *OOO0000000O00O0O0 /340 *OO0OOOO00OOO0O0OO *np .cos (OOOO00OOO0O000OO0 *np .pi /180 ))#line:24

def beamformer (OOO0OO0000O00O000 ,O0O000OOOO0000OOO ,OO0O000OOOOOO0OO0 ,O0O0OOO0OO0O00OOO ):#line:27
    ""#line:35
    O000O00OOO00OO000 ,OOO00O0OOO0O0OOO0 =np .shape (OOO0OO0000O00O000 )#line:38
    OOO00O00O0O0O00O0 =np .arange (0 ,O0O0OOO0OO0O00OOO ,O0O0OOO0OO0O00OOO /OOO00O0OOO0O0OOO0 )#line:41
    O00OO00O00OOOOO0O =np .zeros ((O000O00OOO00OO000 ,1 ),dtype =np .complex_ )#line:49
    OO000O0O0O0O0000O =np .zeros ((len (O0O000OOOO0000OOO ),1 ),dtype =np .complex_ )#line:50
    OO0000OO000OO0OOO =np .fft .fft (OOO0OO0000O00O000 )#line:53
    OO00OO0OOOOOO00O0 =np .abs (OOO00O00O0O0O00O0 -OO0O000OOOOOO0OO0 ).argmin ()#line:57
    O0000OO0O0O0O0OOO =OOO00O00O0O0O00O0 [OO00OO0OOOOOO00O0 ]#line:60
    OO0O0OO0O00OOOO00 =OO0000OO000OO0OOO [:,OO00OO0OOOOOO00O0 ]#line:61
    for O00OOO0O0000000OO ,OOO00OOOOOO0OO000 in enumerate (O0O000OOOO0000OOO ):#line:64
        for OO0O0O00OOO00OOOO in np .arange (0 ,O000O00OOO00OO000 ):#line:66
            OO00OOO00OOO000O0 =beam_filter(O0000OO0O0O0O0OOO ,O000O00OOO00OO000 ,d ,OOO00OOOOOO0OO000 ,OO0O0O00OOO00OOOO )#line:68
            O00OO00O00OOOOO0O [OO0O0O00OOO00OOOO ,:]=OO0O0OO0O00OOOO00 [OO0O0O00OOO00OOOO ]*OO00OOO00OOO000O0 #line:70
        OO000O0O0O0O0000O [O00OOO0O0000000OO ,:]=sum (O00OO00O00OOOOO0O ,1 )#line:72
    O0O0O0OOO0O0OO000 =np .sum (np .square (np .abs (OO000O0O0O0O0000O )),1 )#line:75
    return O0O0O0OOO0O0OO000 #line:77

step  = 8000
estimatedAngle_etu = []
estimatedAngle_prof = []

theta  = np.arange(0,181, 1)
F0 = 480
aroundTheArray=processed.T
for i in range(0, aroundTheArray.shape[1] - step, step):
    buffer = aroundTheArray[:,i:i+step]
    # P_etu  =  myBeamformer(buffer, theta, F0, Fs)
    # estimatedAngle_etu.append(np.argmax(P_etu))
    
    P_prof = beamformer(buffer, theta, F0, Fs)
    estimatedAngle_prof.append(np.argmax(P_prof))

    
t = np.array(range(0, aroundTheArray.shape[1] - step, step))/Fs   

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
    
track=trackTD(filename,width=0.04)

plt.figure()
# plt.plot(t, estimatedAngle_etu, label = 'Our implementation')
plt.plot(t, estimatedAngle_prof, label = 'Beamformer')
plt.plot(track,label = 'pori')
plt.xlabel("time(s)")
plt.ylabel("Angle")
plt.title("Angle estimation with beamformer")   
plt.legend()


# def moving_average(x, w):
#     return np.convolve(x, np.ones(w), 'valid') / w

# filename = 'audio_file.wav'
# data, samplerate = sf.read(filename)

# window_size = 1000
# data_moving_avg = moving_average(data, window_size)


# %%
from collections import Counter
b = Counter(track)
absmax=max(np.abs(track))
final_ang = track[np.argmax(np.abs(track))] if np.abs(b.most_common(1)) is not absmax else b.most_common(1)
print(final_ang)

# %%
