# Import libraries
import numpy as np
import sys 
import pickle
from scipy import signal
import matplotlib.pyplot as plt
#from mapminmax import mapminmax_a
import math
import time
import pandas as pd


"""
def get_predictions(feat,custom_Name,n_epochs,runDir):
    
    # Load Predictive Model
    filepath = runDir+custom_Name+'_LSTM_e'+str(n_epochs)+'.hdf5' # Old model name 
    model = load_model(filepath)
       
    #### Normalise Features    
    # Load corresponding training data
    FileName=runDir+custom_Name+'_data.pkl'
    with open(FileName, 'rb') as f:  # Python 3: open(..., 'rb')
        paramap,_,_,_,_,_,_ = pickle.load(f)
        
    # Normalise test data
    _,x_test = mapminmax_a(paramap, feat,feat)
    
    # Expand dimension for LSTM
    x_test=np.expand_dims(x_test,axis=-1)
    
    
    #### Get Predictions
    model_preds = model.predict(x = x_test)
    class_prediction = np.argmax(model_preds , axis=-1) # Get prediction
    pred_probabilty = np.zeros(np.size(class_prediction))
    for i in range(len(class_prediction)):
        pred_probabilty[i]=model_preds[i,class_prediction[i]] # get the probability of the prediction

    
    return(class_prediction,pred_probabilty)

"""


def getPSD(sr,data,ch_sel,plt) :
    fourier_transform = np.fft.rfft(data)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    
    frequency = np.linspace(0, sr/2, len(power_spectrum))
    if plt==1:
        plt.plot(frequency, power_spectrum[:,ch_sel])
        
    return(frequency, power_spectrum)


def prpr(data_raw,sr,seg_on,seg_duration,filt_on,filter_order,filter_lpass,filter_hpass,plt_pp,ch_sel,banch_sel):
    # Filtering    
    if filt_on==1:
        if filter_hpass>0:
            if filter_lpass>0:
                b, a = signal.butter(filter_order, [2*filter_hpass/sr, 2*filter_lpass/sr],btype='band', analog=False, output='ba', fs=None)
            else:
                b, a = signal.butter(filter_order, 2*filter_hpass/sr,btype='highpass', analog=False, output='ba', fs=None)
                
        else:        
            b, a = signal.butter(filter_order, 2*filter_lpass/sr,btype='lowpass', analog=False, output='ba', fs=None)        

        data_pp = signal.filtfilt(b, a, data_raw,axis=0)
    
    if seg_on==1:
        n_datapoints=len(data_raw)
        tot_duration=n_datapoints/sr
        n_segments=round(tot_duration/seg_duration)
        if filt_on==1:
            data_pp=np.array_split(data_pp, n_segments, axis=0)
            data_raw=np.array_split(data_raw, n_segments, axis=0)
        else:
            data_pp=np.array_split(data_raw, n_segments, axis=0)
            data_raw=np.array_split(data_raw, n_segments, axis=0)
        
        
    if plt_pp==1:
        t_step=1/sr
        n_steps=round(seg_duration/t_step)
        t=np.linspace(0,seg_duration,n_steps)
        fig, axs = plt.subplots(2,2)
        #fig.suptitle('Vertically stacked subplots')
        axs[0,0].plot(t, data_raw[banch_sel][:,ch_sel])
        axs[0,0].set_ylabel('Raw')
        axs[1,0].plot(t, data_pp[banch_sel][:,ch_sel])
        axs[1,0].set_xlabel('Time [s]')
        axs[1,0].set_ylabel('Processed')
        f, Pxx_raw = signal.welch(data_raw[banch_sel][:,ch_sel], sr, nperseg=1024)
        axs[0,1].plot(f,10*np.log10(Pxx_raw))
        axs[0,1].set_ylim([-110, -80])
        axs[0,1].set_ylabel('Power [Db]')

        f, Pxx_pp = signal.welch(data_pp[banch_sel][:,ch_sel], sr, nperseg=1024)
        axs[1,1].plot(f,10*np.log10(Pxx_pp))
        axs[1,1].set_ylim([-110, -80])
        axs[1,1].set_xlabel('Frequency [Hz]')
        axs[1,1].set_ylabel('Power [Db]')
        plt.subplots_adjust(wspace=0.5, hspace=0.5)

    return(data_pp)

def get_features(data_raw,sr):

    from scipy import stats
    import numpy as np
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    plt.ioff()
    import matplotlib.pyplot 
    from scipy import signal
    from scipy.fft import fft
    from tsfresh.feature_extraction import  feature_calculators as ftc
    import math

    def rmsValue(array):
        n = len(array)
        squre = 0.0
        root = 0.0
        mean = 0.0
        
        #calculating Squre
        for i in range(0, n):
            squre += (array[i] ** 2)
        #Calculating Mean
        mean = (squre/ (float)(n))
        #Calculating Root
        root = math.sqrt(mean)
        return root
    
    def nans(data):
        count = 0
        for i in data:
            if not np.isnan(i):
                count += 1
        n_nans=len(data)-count
        return n_nans 
    
    def movingAverage(data,w):
        #k=np.ones([1,w])/w
        y=np.convolve(data, np.ones((w,))/w, mode='same')
        return(y)
    
    def f_FFT(sig,fsamp):
        NFFT=round(len(sig)/10) # take the windowlength of calculation as 10% of the length of the dataset
        OVRLP=round(NFFT/3) # use 30% of the windowlength as overlapping between two windows
        nw = (len(sig)-OVRLP)//(NFFT-OVRLP) # number of windows for calculation
        Wf = signal.hamming(NFFT)                           # window function: hamming
        
        # Define Starting points and end points of data-windows
        startp = np.arange(0,nw,1)*(NFFT-OVRLP)+1     
        endp = np.arange(NFFT,len(sig),NFFT-OVRLP)
        
        # initialize Power-spectrum density matrix
        PSDd = np.zeros([NFFT//2+1,nw])
        
        # Calculate PSD
        for k in range(nw):
            S = sig[startp[k]-1:endp[k]];     #  take a window out of signal
            
            S = np.multiply(S,Wf);                      # apply window function (Hamming) to the signal
            jbjf = fft(S);                  # fourier transform
            
            # adapt fourier transform to power-spectrum density, PSD
            jbjf[1:] = 2*jbjf[1:] 
            jbjf = np.divide(jbjf,NFFT)
            PSDd[:,k] = np.multiply(jbjf[0:NFFT//2+1], np.conj(jbjf[0:NFFT//2+1])  )
        
        # Get average PSD
        psd=np.mean(PSDd, axis=1)
        
        # construct the corresponding frequency values
        freq = np.arange(0,(NFFT-1)/NFFT*fsamp,fsamp/NFFT);
        fVals = freq[1:NFFT//2+1];
        return(psd,fVals)
    
    def f_FFTsimple(sig,fsamp):
        
        jbjf = fft(sig);                  # fourier transform
        
        # adapt values for power-spectrum density, PSD
        jbjf[1:] = 2*jbjf[1:]
        jbjf = np.divide(jbjf,len(sig))
        psd = np.real(np.multiply(jbjf[0:len(sig)//2+1],np.conj(jbjf[0:len(sig)//2+1])))
        
        # construct the corresponding frequency values  
        freq = np.arange(0,(len(sig)-1)/len(sig)*fsamp,fsamp/len(sig))
        fVals = freq[1:len(sig)//2+1];
        
        return(psd,fVals)
       
    def f_getEntr(x):
        x=np.divide(x,np.sum(x))
        log2vect=np.zeros(np.size(x))
        for k in range(len(x)):
            if x[k]==0:
                log2vect[k]=0
            else:
                    log2vect[k]=math.log2(x[k])
                    
        Entropy=-1*np.sum(np.multiply(log2vect,x))
        
        return(Entropy)
    
    def f_findmatch(Array,TargetVal):
        # find closest match of a value in an array
        dummy = abs(Array-TargetVal)     #absolute value of difference between array values and target   
        Index=np.argmin(dummy)
        return Index
    
    ### INPUT
    if data_raw.ndim<2:
        data=np.zeros([len(data_raw),1])
        data[:,0]=data_raw
    else:
        data=data_raw
    
    # AR Features
    AR_order=4
    
    # Moving Average features
    STAdur = sr;    #number of datasamples for calculation the short-term averagege (1 second)
    MTAdur = sr*20;   #number of datasamples for calculation the medium-term averagege (20 seconds)
    LTAdur = sr*120;  #number of datasamples for calculation the long-term averagege (2 minutes)
    n_bins=26
    
    # Frequency Domain features
    fsamp=1
    ### 
    
    #Preallocation
    n_ch=data.shape[1]
    d_zeros=np.zeros(n_ch)    
    d_mean=np.zeros(n_ch)
    d_median=np.zeros(n_ch)
    d_mode=np.zeros(n_ch)
    d_nans=np.zeros(n_ch)
    d_std=np.zeros(n_ch)
    d_var=np.zeros(n_ch)
    d_skew=np.zeros(n_ch)
    d_kurt=np.zeros(n_ch)
    d_ar1=np.zeros(n_ch)
    d_ar2=np.zeros(n_ch)
    d_ar3=np.zeros(n_ch)
    VarSTAdist=np.zeros(n_ch)
    VarMTAdist=np.zeros(n_ch)
    VarLTAdist=np.zeros(n_ch)
    HomogSTAdist=np.zeros(n_ch)
    HomogMTAdist=np.zeros(n_ch)
    HomogLTAdist=np.zeros(n_ch)
    Entr1=np.zeros(n_ch)
    Entr2=np.zeros(n_ch)
    Kurt1=np.zeros(n_ch)
    Kurt2=np.zeros(n_ch)
    Homog1=np.zeros(n_ch)
    Homog2=np.zeros(n_ch)
    PSDrange1=np.zeros(n_ch)
    PSDrange2=np.zeros(n_ch)
    JBJFrange1=np.zeros(n_ch)
    JBJFrange2=np.zeros(n_ch)
    
    f_abs_en=np.zeros(n_ch)
    f_abs_sc=np.zeros(n_ch)
    f_ac=np.zeros(n_ch)
    f_bc=np.zeros(n_ch)
    f_c3=np.zeros(n_ch)
    f_cid_ce=np.zeros(n_ch)
    f_lstd=np.zeros(n_ch)
    f_lsam=np.zeros(n_ch)
    f_lsbm=np.zeros(n_ch)
    f_abs_mc=np.zeros(n_ch)
    f_mc=np.zeros(n_ch)
    f_m2d=np.zeros(n_ch)
    f_n_peaks=np.zeros(n_ch)
    f_ac_part=np.zeros(n_ch)
    f_nu=  np.zeros(n_ch)
    f_nu2= np.zeros(n_ch)
    f_rbs= np.zeros(n_ch)
    f_rms= np.zeros(n_ch)
    f_srv= np.zeros(n_ch)
    f_sl= np.zeros(n_ch)
    f_fulch= np.zeros(n_ch)
    f_count0= np.zeros(n_ch)
    f_vlstd= np.zeros(n_ch)
    d_cov= np.zeros(n_ch)
    
    
    
    
    for i in range(n_ch):  
        # Exclude NaNs
        nan_array = np.isnan(data[:,i])
        not_nan_array = ~ nan_array # data wo NaNs
        data_ch_noNans = data[not_nan_array,i] 
        dt_len=len(data_ch_noNans)
        lag=min(round(dt_len/100),3000)
        
        if len(data_ch_noNans)!=0:

            ### 1. TIME DOMAIN
            #t = time.time()
            # 1.1 Descriptive Statistics
            d_mean[i]=np.mean(data_ch_noNans)
            d_median[i]=np.median(data_ch_noNans)  # Median 
            d_mode[i]=stats.mode(data_ch_noNans)[0]# Most common value       
            d_std[i]=np.std(data_ch_noNans)
            d_var[i]=np.var(data_ch_noNans)
            d_skew[i]=stats.skew(data_ch_noNans)
            d_kurt[i]=stats.kurtosis(data_ch_noNans, fisher=False)
            d_cov[i]=ftc.variation_coefficient(data_ch_noNans) # coefficient of variation
                     

            # 1.2 Length, repetitions etc 
            d_zeros[i]=ftc.value_count(data_ch_noNans, 0)/len(data_ch_noNans) # Number of 0s normalised to lenght of the dataset [9] 
            d_nans[i]=nans(data[:,i])/len(data_ch_noNans) # Number of Nans normalised to lenght of the dataset            
            f_lsam[i]=ftc.longest_strike_above_mean(data_ch_noNans) # Get length of longest strike above mean
            f_lsbm[i]=ftc.longest_strike_below_mean(data_ch_noNans) # Get length of longest strike below mean            
            f_n_peaks[i]=ftc.number_peaks(data_ch_noNans,lag) # Number of peaks in lag
            f_srv[i]=ftc.sum_of_reoccurring_values(data_ch_noNans) # Returns the sum of all values, that are present in the time series more than once.
            f_count0[i]=ftc.value_count(data_ch_noNans, 0)/len(data_ch_noNans) # count 0
            f_nu[i]=  ftc.percentage_of_reoccurring_datapoints_to_all_datapoints(data_ch_noNans) # percentage of non-unique data points. Non-unique means that they are contained another time in the time series again.
            f_nu2[i]= ftc.ratio_value_number_to_time_series_length(data_ch_noNans) # # unique values / # values            

            # 1.3 Autocorrelations, Complexity metrics etc.  
            f_ac[i]=ftc.autocorrelation(data_ch_noNans,lag) # Autocorrelation with a lag  [18]
            f_ac_part[i]=ftc.partial_autocorrelation(data_ch_noNans,[{"lag":lag}])[0][1] #  value of the partial autocorrelation function at the given lag
            f_bc[i]=ftc.benford_correlation(data_ch_noNans)
            f_c3[i]=ftc.c3(data_ch_noNans,lag) # nonlinearity metric
            #f_cid_ce[i]=ftc.cid_ce(data_ch_noNans, "False") # Complexity metric
            f_sl[i]=int(ftc.symmetry_looking(data_ch_noNans, [{"r": 0.01}])[0][1]==True) # Symmetric or not 22
            f_fulch[i]=ftc.time_reversal_asymmetry_statistic(data_ch_noNans, lag) 

            # 1.4 Sums and statistic combinations # 24
            f_rms[i]=rmsValue(data_ch_noNans) # RMS of signal [24]
            #f_abs_en[i]=ftc.abs_energy(data_ch_noNans) #  absolute energy of the time series which is the sum over the squared values           
            #f_abs_sc[i]=ftc.absolute_sum_of_changes(data_ch_noNans)#  absolute value of consecutive changes in the series x
            f_abs_mc[i]=ftc.mean_abs_change(data_ch_noNans) # mean over the absolute differences between subsequent time series values
            f_mc[i]=ftc.mean_change(data_ch_noNans) # mean over the differences between subsequent time series values
            f_m2d[i]=ftc.mean_second_derivative_central(data_ch_noNans) #  mean value of a central approximation of the second derivative
            f_vlstd[i]=int(ftc.variance_larger_than_standard_deviation(data_ch_noNans)==True) # Variance larger than 1   29
            f_lstd[i]=int(ftc.large_standard_deviation(data_ch_noNans, 0.25)==True) # Checks if std is larger than 25 % of the range
            f_rbs[i]=ftc.ratio_beyond_r_sigma(data_ch_noNans,r=1) # Ratio of values that are more than r*std(x) (so r sigma) away from the mean of x.
           # print(time.time() - t)
            # 1.5 AR Parameters 
            # try:
            #     rho, _ = sm.regression.yule_walker(data_ch_noNans, order=AR_order)
            #     d_ar1[i]=rho[0]
            #     d_ar2[i]=rho[1]
            #     d_ar3[i]=rho[2]
            # except:
            #     pass
            #print(time.time() - t)
                        
            # 1.6 Moving Average features normalised to mean
            sta=movingAverage(data_ch_noNans,STAdur)/np.mean(data_ch_noNans)  
            mta=movingAverage(data_ch_noNans,MTAdur)/np.mean(data_ch_noNans)
            lta=movingAverage(data_ch_noNans,LTAdur)/np.mean(data_ch_noNans)              
            # Get the distribution of the short-, medium-, and long-term moving average values
            fig = matplotlib.figure.Figure()
            ax = matplotlib.axes.Axes(fig, (0,0,0,0))
            # NEW: to prevent error
            if np.any(np.isinf(sta)):
                sta = sta[~np.isinf(sta)]
            if np.any(np.isinf(mta)):
                mta = mta[~np.isinf(mta)]
            if np.any(np.isinf(lta)):
                lta = lta[~np.isinf(lta)]

            Pshort = ax.hist(sta,bins=n_bins)[0]
            del ax, fig            
            fig = matplotlib.figure.Figure()
            ax = matplotlib.axes.Axes(fig, (0,0,0,0))
            Pmedium = ax.hist(mta,bins=n_bins)[0]
            del ax, fig
            fig = matplotlib.figure.Figure()
            ax = matplotlib.axes.Axes(fig, (0,0,0,0))
            Plong = plt.hist(lta,bins=n_bins)[0]
            del ax, fig            
            # Variance
            VarSTAdist[i] = np.std(Pshort)/np.mean(Pshort)  # 30
            VarMTAdist[i] = np.std(Pmedium)/np.mean(Pmedium)
            VarLTAdist[i] = np.std(Plong)/np.mean(Plong)            
            # Homogenity of distribution (maximum value compared to mean value)
            HomogSTAdist[i] = Pshort.max()/np.mean(Pshort)
            HomogMTAdist[i] = Pmedium.max()/np.mean(Pmedium)
            HomogLTAdist[i] = Plong.max()/np.mean(Plong);    
            #print(time.time() - t)
        
            #### 2. FREQUENCY DOMAIN
            # 2.1 
            # calculate the frequency spectrum using windows
            psd,fVals=f_FFT(data_ch_noNans-np.mean(data_ch_noNans),fsamp)              
            # calculate the frequency spectrum using the entire dataset 
            jbjf,fVals2 = f_FFTsimple(data_ch_noNans-np.mean(data_ch_noNans),fsamp)            
            # Get features
            #Entr1[i]=f_getEntr(np.divide(psd,psd.max())) # [36]
            #Entr2[i]=f_getEntr(np.divide(jbjf,jbjf.max()))
            Kurt1[i] = stats.kurtosis(np.divide(psd,psd.max()))
            Kurt2[i] = stats.kurtosis(np.divide(jbjf,jbjf.max()))
            Homog1[i] = np.divide(np.divide(psd,psd.max()).max(),   np.mean(np.divide(psd,psd.max())))
            Homog2[i] = np.divide(np.divide(jbjf,jbjf.max()).max(),   np.mean(np.divide(jbjf,jbjf.max())))            
            #print(time.time() - t)
            # 2.2 Cumulative / Ranges 
            # Get the cumulated frequency domain spectrum
            try:
                PSDcumul = np.cumsum(psd)                   
                # Get range of frequency containing 95% of energy
                Index_1 = f_findmatch(PSDcumul,0.025*PSDcumul[-1])
                Index_2 = f_findmatch(PSDcumul,0.975*PSDcumul[-1])
                Index_2 = np.minimum(Index_2,len(fVals)-1)
                PSDrange1[i] = fVals[Index_2] - fVals[Index_1]            
                # Get range of frequency containing 50% of energy
                Index_1 = f_findmatch(PSDcumul,0.25*PSDcumul[-1])
                Index_2 = f_findmatch(PSDcumul,0.75*PSDcumul[-1])
                PSDrange2[i] = fVals[Index_2] - fVals[Index_1]            
                
                # Get the cumulated frequency domain spectrum (over entire dataset at once)   
                JBJFcumul = np.cumsum(jbjf);            
                # Get range of frequency containing 95% of energy
                Index_1 = f_findmatch(JBJFcumul,0.025*JBJFcumul[-1])
                Index_2 = f_findmatch(JBJFcumul,0.975*JBJFcumul[-1])
                Index_2 = np.minimum(Index_2,len(fVals2)-1)
                JBJFrange1[i] = fVals2[Index_2] - fVals2[Index_1]            
                # Get range of frequency containing 50% of energy
                Index_1 = f_findmatch(JBJFcumul,0.25*JBJFcumul[-1])
                Index_2 = f_findmatch(JBJFcumul,0.75*JBJFcumul[-1])
                JBJFrange2[i] = fVals2[Index_2] - fVals2[Index_1]
                #print(time.time() - t)
            except:
                    pass
                
    # 1.1 Descriptive Statistics [8]
    feat_11=np.concatenate((d_mean[:,None],d_median[:,None],d_mode[:,None],d_std[:,None],d_var[:,None],d_skew[:,None],d_kurt[:,None],d_cov[:,None]),axis=1)  
    # 1.2 Length, repetitions etc [9]
    feat_12=np.concatenate((d_zeros[:,None],d_nans[:,None],f_lsam[:,None],f_lsbm[:,None],f_n_peaks[:,None],f_srv[:,None],f_count0[:,None],f_nu[:,None],f_nu2[:,None]),axis=1)  
    # 1.3 Autocorrelations, Complexity metrics etc. [7]
    feat_13=np.concatenate((f_ac[:,None],f_ac_part[:,None],f_bc[:,None],f_c3[:,None],f_cid_ce[:,None],f_sl[:,None],f_fulch[:,None]),axis=1)  
    # 1.4 Sums and statistic combinations [9]
    feat_14=np.concatenate((f_rms[:,None],f_abs_en[:,None],f_abs_sc[:,None],f_abs_mc[:,None],f_mc[:,None],f_m2d[:,None],f_vlstd[:,None],f_lstd[:,None],f_rbs[:,None]),axis=1)               
    # 1.5 AR Parameters [3]
    feat_15=np.concatenate((d_ar1[:,None],d_ar2[:,None],d_ar3[:,None]),axis=1)
    # 1.6 Moving Average features normalised to mean   [6]
    feat_16=np.concatenate((VarSTAdist[:,None],VarMTAdist[:,None],VarLTAdist[:,None],HomogSTAdist[:,None],HomogMTAdist[:,None],HomogLTAdist[:,None]),axis=1)
    # 2.1     [6] starting from 40
    feat_21=np.concatenate((Entr1[:,None],Entr2[:,None],Kurt1[:,None],Kurt2[:,None],Homog1[:,None],Homog2[:,None]),axis=1)  
    # 2.2 Cumulative / Ranges [4]
    feat_22=np.concatenate((PSDrange1[:,None],PSDrange2[:,None],JBJFrange1[:,None],JBJFrange2[:,None]),axis=1)  

    # Assembel features together
    #feat_all=np.concatenate((feat_11,feat_12,feat_13,feat_14,feat_15,feat_16,feat_21,feat_22),axis=1)
    feat_all=np.concatenate((feat_11,feat_12,feat_13,feat_14,feat_16,feat_21,feat_22),axis=1)
    #print(time.time() - t)

    return (feat_all)




def plot_confusion_matrix(y_actu, y_pred,plt_on=1,title='CM'):
    cmap=plt.cm.gray_r
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'])
    df_confusion=df_confusion.rename(columns={0: "Normal", 1: "Uncertain", 2:"Abnormal"})
    df_confusion=df_confusion.rename(index={0: "Normal",2:"Abnormal"})
    df_conf_norm = df_confusion.divide(df_confusion.sum(axis=1),axis=0)
    
    plt.matshow(df_conf_norm, cmap=cmap) # imshow
    for (i, j), z in np.ndenumerate(df_conf_norm):
        plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',color='red',weight="bold")

    #plt.title(title)
    #plt.colorbar()
    plt.rcParams["font.family"] = "Times New Roman"
    tick_marks = np.arange(len(df_conf_norm.columns))
    plt.xticks(tick_marks, df_conf_norm.columns, rotation=45)
    plt.yticks(tick_marks, df_conf_norm.index)
    plt.ylabel(df_conf_norm.index.name)
    plt.xlabel(df_conf_norm.columns.name)
    if plt_on==1:
        aa=title+'.png'
        plt.savefig(aa, format = 'png', dpi=300, bbox_inches = 'tight')
        
def plot_confusion_matrix_01(y_actu, y_pred,plt_on=1,title='CM'):
    cmap=plt.cm.gray_r
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'])
    df_confusion=df_confusion.rename(columns={0: "Healthy", 1:"Fault"})
    df_confusion=df_confusion.rename(index={0: "Healthy",1:"Fault"})
    df_conf_norm = df_confusion.divide(df_confusion.sum(axis=1),axis=0)
    
    plt.matshow(df_conf_norm, cmap=cmap) # imshow
    for (i, j), z in np.ndenumerate(df_conf_norm):
        plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',color='red',weight="bold")

    #plt.title(title)
    #plt.colorbar()
    plt.rcParams["font.family"] = "Times New Roman"
    tick_marks = np.arange(len(df_conf_norm.columns))
    plt.xticks(tick_marks, df_conf_norm.columns, rotation=45)
    plt.yticks(tick_marks, df_conf_norm.index)
    plt.ylabel(df_conf_norm.index.name)
    plt.xlabel(df_conf_norm.columns.name)
    if plt_on==1:
        aa=title+'.png'
        plt.savefig(aa, format = 'png', dpi=300, bbox_inches = 'tight')
        
def plot_confusion_matrix_012(y_actu, y_pred,plt_on=1,title='CM'):
    cmap=plt.cm.gray_r
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['OCC'], colnames=['XGBoost']) # Actual , Predicted
    df_confusion=df_confusion.rename(columns={0: "Normal", 1: "Uncertain", 2:"Abnormal"})
    df_confusion=df_confusion.rename(index={0: "Normal", 1: "Uncertain", 2:"Abnormal"})
    df_conf_norm = df_confusion.divide(df_confusion.sum(axis=1),axis=0)
    plt.figure(figsize=(4.5,3))
    plt.matshow(df_conf_norm, cmap=cmap) # imshow
    for (i, j), z in np.ndenumerate(df_conf_norm):
        plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',color='red',weight="bold")

    #plt.title(title)
    #plt.colorbar()
    plt.rcParams["font.family"] = "Times New Roman"
    tick_marks = np.arange(len(df_conf_norm.columns))
    plt.xticks(tick_marks, df_conf_norm.columns, rotation=45)
    plt.yticks(tick_marks, df_conf_norm.index)
    plt.ylabel(df_conf_norm.index.name)
    plt.xlabel(df_conf_norm.columns.name)
    if plt_on==1:
        aa=title+'.png'
        plt.savefig(aa, format = 'png', dpi=300, bbox_inches = 'tight')