#%%
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from scipy import signal
import soundfile as sf
#%%
# These values can be adapted according to your requirements.
samplerate = 48000
seconds = 5
downsample = 1
input_gain_db = 12
device = 'snd_rpi_i2s_card'

def butter_highpass(cutoff, fs, order=5):
    '''
    Helper function for the high-pass filter.
    '''
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    '''
    High-pass filter for digital audio data.
    '''
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def set_gain_db(audiodata, gain_db):
    '''
    This function allows to set the audio gain
    in decibel. Values above 1 or below -1 are set to
    the max/min values.
    '''
    audiodata *= np.power(10, gain_db/10)
    return np.array([1 if s > 1 else -1 if s < -1 else s for s in audiodata], dtype=np.float32)

def process_audio_data(audiodata):
    # Extract mono channels from input data.
    ch1 = np.array(audiodata[::downsample, 0], dtype=np.float32)
    ch2 = np.array(audiodata[::downsample, 1], dtype=np.float32)

    # High-pass filter the data at a cutoff frequency of 10Hz.
    # This is required because I2S microhones have a certain DC offset
    # which we need to filter in order to amplify the volume later.
    ch1 = butter_highpass_filter(ch1, 40, samplerate)
    ch2 = butter_highpass_filter(ch2, 40, samplerate)

    # Amplify audio data.
    # Recommended, because the default input volume is very low.
    # Due to the DC offset this is not recommended without using
    # a high-pass filter in advance.
    ch1 = set_gain_db(ch1, input_gain_db)
    ch2 = set_gain_db(ch2, input_gain_db)

    # Output the data in the same format as it came in.
    return np.array([[ch1[i], ch2[i]] for i in range(len(ch1))], dtype=np.float32)

# Record stereo audio data for the given duration in seconds.
rec = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=2)
# Wait until the recording is done
sd.wait()  # Wait until recording is finished

# Process the audio data as explained above.
processed = process_audio_data(rec)

#write('output.wav', samplerate, rec.astype(np.int16)) 


# Write the processed audio data to a wav file.
write('out.wav', int(samplerate/downsample), processed)
sf.write('locus_90a90.wav', processed, samplerate, subtype='PCM_16')

#%%
import matplotlib.pyplot as plt
import ipywidgets as widgets
import numpy as np
import wave
import librosa
# from client import array
import time


BLK = 480000
N = 2
d = 0.04
#%%
filename='mollcricket_45a90.wav'
with wave.open(filename, 'rb') as f:
    frames = f.readframes(-1)
    audio = np.frombuffer(frames, dtype='int16')

y_stereo,sr  = librosa.load(filename, mono=False)
Fs = sr
#%%
processed=y_stereo.T
#%%


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

#%%
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
def beam_filter_etu(array, freq_vector, theta0=0, mic_nb: int = 0):
    """Compute the filter frequency response of a DSB beamformer for one microphone

    Args:
        array (array_server obj): array structure controlling the acquisition system.
        freq_vector (np.array): frequency vector. 
        theta0 (int, optional): focusing angular direction (in degrees). Defaults to 0.
        mic_id (int, optional): microphone id. Defaults to 0.

    Returns:
        np.array: the filter frequency response. Shape is (len(freq_vector),).
    """

    N = 2
    d = 0.04
    # Microphone position x
    z = (mic_nb - (N +1)/2) * d
    # Filter's frequency response
    return np.exp (-2j *np.pi *freq_vector/340 * z *np.cos(theta0 *np.pi /180 ))
# %%
f = np.arange(0, 5000, Fs/len(processed))
plt.figure(figsize = (10,4))
microphones = [0,1]
for i in microphones:
    frequency_response = beam_filter_etu(None, f, 0, i)
    plt.plot(f, frequency_response, label = "micro {}".format(i))
plt.title("Frequency responses obtained for two filters associated to two\n different microphone outputs when $\Theta_0 = 0 $")
plt.xlabel("Frenquency(hz)")
plt.legend(loc='upper left')
plt.ylabel("Amplitude")
plt.show()
# %%
f = np.arange(0, 10000, Fs/len(processed))
plt.figure(figsize = (10,4))
microphones = [0,1]
for i in microphones:
    frequency_response = beam_filter_etu(None, f, 90, i)
    plt.plot(f, frequency_response, label = "micro {}".format(i))
plt.title("Frequency responses obtained for two filters associated to two\n different microphone outputs when $\Theta_0 = 90 $")
plt.xlabel("Frenquency(hz)")
plt.legend(loc='upper left')
plt.ylabel("Amplitude")
plt.show()
# %%
M_fft = np.fft.fft(processed)

plt.figure(figsize = (10,4))
for indx in range(2):
    plot_fft(processed, microphone = indx,  plot_absFFT_only = True)

# %%
# Compute K0 as the list index of the maximum value of abs(fft)
F0 = 350
M7 = M_fft[:,1]
k0 = np.argmax(np.abs(M7[0:BLK//2]))
k02 = (np.abs(f - F0))

k03 = np.min(np.argmax(np.abs(M_fft[0:BLK//2,:]), axis = 1 ))

print("k0 = ", k0)
print("La fréquence à k0 : f[k0] =", f[k0]) 

# %%
M = []
for i in range(2):
    M.append(M_fft[k0][i]) 
print("M = ")
for complexValue in M:
     print(complexValue)
# %%
filter_outputs_at_theta0 = []
for i in range(2):
    val = M[i] *  beam_filter_etu(None, f[k0], theta0 = 0, mic_nb = i)
    filter_outputs_at_theta0.append(val) 
    

print("filter_outputs_at_theta0 = ")
for complexValue in filter_outputs_at_theta0:
    print(complexValue)

# %%
Y_theta0 = np.sum(filter_outputs_at_theta0)
print("beamformer output : \nY_theta0 = ", Y_theta0)

P_theta0 = np.abs(Y_theta0)**2
print("P_theta0 = ", P_theta0)
# %%
filter_outputs_theta120 = []
for i in range(2):
    val = M[i] *  beam_filter_etu(None, f[k0], 120 , mic_nb = i)
    filter_outputs_theta120.append(val) 
    
Y_theta120 = np.sum(filter_outputs_theta120)
print("Y_theta90 = ", Y_theta120)

P_theta120 = np.abs(Y_theta120)**2
print("P_theta00 = ", P_theta120)
# %%
thetaTab = np.arange(0, 181, 1)
power = []
for theta in thetaTab:
    filter_outputs = []
    for i in range(2):
        val = M[i] *  beam_filter_etu(None, f[k0], theta , mic_nb = i)
        filter_outputs.append(val) 
    Y = np.sum(filter_outputs)
    P = np.abs(Y)**2
    power.append(P)
    
plt.figure(figsize = (10,4))
plt.plot(thetaTab,power)
plt.title(" Beamformer Output power P as a function of the angle $\Theta_0$ ")
plt.xlabel("$\Theta$(deg)")
plt.ylabel("Beamformer Output power")
plt.grid(visible=True, which='both', color='0.65', linestyle='-')
plt.minorticks_on()
# %%
print("The Angle corresponding to the highest computed Power :", np.argmax(power))
# %%
def myBeamformer(buffer, theta , F0 , Fs):
    """ Compute the energy map from a Delay -And -Sum beamforming
    Args :
    buffer (np. array ): audio buffer , of size N x BLK_SIZE
    theta (np. array ): array of angular value in degrees listing the polarization angle of the beamformer
    F0 ( float ): source frequency to localize
    Fs ( float ): sampling frequency
    """
    N, BLK = np.shape(buffer)
    M_fft = np.fft.fft(buffer)
    f = np.arange(0, Fs, Fs/BLK)
    
    #k0 = np.min(np.argmax(np.abs(M_fft[:,0:BLK//2]), axis = 1 ))
    k0 = (np.argmin(np.abs(f - F0)))
    
    M = M_fft[:,:k0]
    power = []    
    for myTheta in theta:
        W = []
        for i in range(N):
            W.append(beam_filter_etu(None, f[k0], theta0 = myTheta, mic_nb= i))
         
        # Beamformer Ouput
        Y = M*W
        # Compute Power coresponding to theta
        P = np.abs(np.sum(Y))**2
        power.append(P)
        
    return power
#%%
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
# if __name__ =="__main__":#line:79
#     print ("Simulation")#line:80
#     beamformer (1 )
# %%
theta = np.arange(0,180, 1)
# %%
# P400hz  = myBeamformer(processed, theta, 300, Fs)
# P=list(P400hz)
# # %%
# plt.figure(figsize=(10,8))
# plt.suptitle("Energy Maps corresponding to differents frequencies F0")
# labels = ['400 Hz']
# colors = ['green',] 
# for i in range(1):
#     plt.subplot(2,4,i+1)
#     plt.plot(theta, P[i][0:180], color=colors[i], label = labels[i])
#     plt.legend(loc='upper right')
#     plt.xlabel("$\Theta$(deg)")
#     if i == 0 : plt.ylabel("Beamformer Output power")
#     plt.grid(visible=True, which='both', color='0.95', linestyle='-')
#     plt.minorticks_on()
    
    
# plt.suptitle("Energy Maps corresponding to differents frequencies F0")
# for i in range(len(P)):
#     plt.subplot(2,4,5+i, polar = True)
#     plt.polar(theta*np.pi/180, P[i], color=colors[i], label = labels[i])
#     plt.legend()
# %%
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
    
plt.figure()
# plt.plot(t, estimatedAngle_etu, label = 'Our implementation')
plt.plot(t, estimatedAngle_prof, label = 'profs implementation')
plt.xlabel("temps (s)")
plt.ylabel("Angle en degré")
plt.title("Angle estimé en fonction du temps")   
plt.legend()
# %%
import librosa
import numpy as np
import matplotlib.pyplot as plt

def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None, mono=False)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    stft = np.abs(librosa.stft(audio))
    spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
    return audio, spectrogram, sr
# %%

def estimate_azimuth_angle(stereo_audio, sr):
    cross_correlation = np.correlate(stereo_audio[0], stereo_audio[1], mode='full')
    peak_index = np.argmax(cross_correlation)
    delay = (peak_index - len(stereo_audio[0]) + 1) / sr
    speed_of_sound = 343.2
    distance = speed_of_sound * delay
    radius = 1.0
    azimuth_angle = np.arccos(distance / radius)
    azimuth_degrees = np.degrees(azimuth_angle)
    return azimuth_degrees

def estimate_distance(stereo_audio, sr):
    intensity_left = np.mean(stereo_audio[0]**2)
    intensity_right = np.mean(stereo_audio[1]**2)
    average_intensity = (intensity_left + intensity_right) / 2.0
    reference_distance = 1.0
    distance = reference_distance * np.sqrt(average_intensity)
    return distance
# %%
def plot_sound_source(distance, azimuth_degrees):
    azimuth_radians = np.radians(azimuth_degrees)
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection='polar')
    ax.scatter(azimuth_radians, distance, color='red', s=100, label='Sound Source')
    ax.set_rticks([0.5, 1.0, 1.5, 2.0])
    ax.set_rlabel_position(45)
    ax.set_title('Sound Source Location', pad=20)
    ax.legend(loc='upper right')
    plt.show()
# %%
file_path = 'stereo_file.wav'

# Preprocess the audio and obtain the audio waveform and spectrogram
audio, spectrogram, sr = preprocess_audio(file_path)
#%%
# Estimate the azimuth angle of the sound source
azimuth_degrees = estimate_azimuth_angle(audio, sr)
#%%
# Estimate the distance of the sound source
distance = estimate_distance(audio, sr)
#%%
# Plot the sound source location
plot_sound_source(distance, azimuth_degrees)
# %%
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load('stereo_file.wav')
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                    fmax=8000)
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S,
                                             ref=np.max),
                          y_axis='mel', fmax=8000,
                          x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.tight_layout()
plt.show()
# %%
