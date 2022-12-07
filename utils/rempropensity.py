import scipy.signal
import numpy as np
import scipy.io as so
import os.path
import re
import sys
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
from sklearn.mixture import GaussianMixture

######## Functions in Sleepy #########
def set_fontsize(fs):
    import matplotlib
    matplotlib.rcParams.update({'font.size': fs})

def set_fontarial():
    """
    set Arial as default font
    """
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = "Arial"

def box_off(ax):
    """
    similar to Matlab's box off
    """
    ax.spines["top"].set_visible(False)    
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  

def sleep_example(ppath, name, tlegend, tstart, tend, fmax=30, fig_file='', vm=[], ma_thr=10,
                  fontsize=12, cb_ticks=[], emg_ticks=[], r_mu = [10, 100], fw_color=True):
    """
    plot sleep example
    :param ppath: base folder
    :param name: recording name
    :param tstart: start (in seconds) of shown example interval
    :param tend: end of example interval
    :param tlegend: length of time legend
    :param fmax: maximum frequency shown for EEG spectrogram
    :param fig_file: file name where figure will be saved
    :param vm: saturation of EEG spectrogram
    :param fontsize: fontsize
    :param cb_ticks: ticks for colorbar
    :param emg_ticks: ticks for EMG amplitude axis (uV)
    :param r_mu: range of frequencies for EMG amplitude
    :param fw_color: if True, use standard color scheme for brainstate (gray - NREM, violet - Wake, cyan - REM);
            otherwise use Shinjae's color scheme
    """
    set_fontarial()
    set_fontsize(fontsize)

    # True, if laser exists, otherwise set to False
    plaser = False

    sr = get_snr(ppath, name)
    nbin = np.round(2.5 * sr)
    dt = nbin * 1 / sr

    istart = int(np.round(tstart/dt))
    iend   = int(np.round(tend/dt))
    dur = (iend-istart+1)*dt

    M,K = load_stateidx(ppath, name)
    #kcut = np.where(K>=0)[0]
    #M = M[kcut]
    if tend==-1:
        iend = len(M)
    M = M[istart:iend]

    seq = get_sequences(np.where(M==2)[0])
    for s in seq:
        if len(s)*dt <= ma_thr:
            M[s] = 0

    t = np.arange(0, len(M))*dt

    P = so.loadmat(os.path.join(ppath, name, 'sp_%s.mat' % name), squeeze_me=True)
    SPEEG = P['SP']#/1000000.0
    # calculate median for choosing right saturation for heatmap
    med = np.median(SPEEG.max(axis=0))
    if len(vm) == 0:
        vm = [0, med*2.5]
    #t = np.squeeze(P['t'])
    freq = P['freq']
    P = so.loadmat(os.path.join(ppath, name, 'msp_%s.mat' % name), squeeze_me=True)
    SPEMG = P['mSP']#/1000000.0


    # create figure
    plt.ion()
    plt.figure(figsize=(8,4))

    # axis in the background to draw laser patches
    axes_back = plt.axes([0.1, .4, 0.8, 0.52])
    axes_back.get_xaxis().set_visible(False)
    axes_back.get_yaxis().set_visible(False)
    axes_back.spines["top"].set_visible(False)
    axes_back.spines["right"].set_visible(False)
    axes_back.spines["bottom"].set_visible(False)
    axes_back.spines["left"].set_visible(False)

    plt.ylim((0,1))
    plt.xlim([t[0], t[-1]])


    # show brainstate
    axes_brs = plt.axes([0.1, 0.4, 0.8, 0.05])
    cmap = plt.cm.jet
    if fw_color:
        my_map = cmap.from_list('brs', [[1, 118./255, 245./255], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 4)
    else:
        my_map = cmap.from_list('brs', [[0, 0, 0], [153 / 255.0, 76 / 255.0, 9 / 255.0],
                                        [120 / 255.0, 120 / 255.0, 120 / 255.0], [1, 0.75, 0]], 4)

    tmp = axes_brs.pcolorfast(t, [0, 1], np.array([M]), vmin=0, vmax=3)
    tmp.set_cmap(my_map)
    axes_brs.axis('tight')
    axes_brs.axes.get_xaxis().set_visible(False)
    axes_brs.axes.get_yaxis().set_visible(False)
    axes_brs.spines["top"].set_visible(False)
    axes_brs.spines["right"].set_visible(False)
    axes_brs.spines["bottom"].set_visible(False)
    axes_brs.spines["left"].set_visible(False)

    axes_legend = plt.axes([0.1, 0.33, 0.8, 0.05])
    plt.ylim((0,1.1))
    plt.xlim([t[0], t[-1]])
    plt.plot([0, tlegend], [1, 1], color='black')
    plt.text(tlegend/4.0, 0.1, str(tlegend) + ' s')
    axes_legend.spines["top"].set_visible(False)
    axes_legend.spines["right"].set_visible(False)
    axes_legend.spines["bottom"].set_visible(False)
    axes_legend.spines["left"].set_visible(False)
    axes_legend.axes.get_xaxis().set_visible(False)
    axes_legend.axes.get_yaxis().set_visible(False)

    # show spectrogram
    ifreq = np.where(freq <= fmax)[0]
    # axes for colorbar
    axes_cbar = plt.axes([0.82, 0.68, 0.1, 0.2])
    # axes for EEG spectrogram
    axes_spec = plt.axes([0.1, 0.68, 0.8, 0.2], sharex=axes_brs)
    im = axes_spec.pcolorfast(t, freq[ifreq], SPEEG[ifreq, istart:iend], cmap='jet', vmin=vm[0], vmax=vm[1])
    axes_spec.axis('tight')
    axes_spec.set_xticklabels([])
    axes_spec.set_xticks([])
    axes_spec.spines["bottom"].set_visible(False)
    plt.ylabel('Freq (Hz)')
    box_off(axes_spec)
    plt.xlim([t[0], t[-1]])

    # colorbar for EEG spectrogram
    cb = plt.colorbar(im, ax=axes_cbar, pad=0.0, aspect=10.0)
    cb.set_label('Power ($\mathrm{\mu}$V$^2$s)')
    if len(cb_ticks) > 0:
        cb.set_ticks(cb_ticks)
    axes_cbar.set_alpha(0.0)
    axes_cbar.spines["top"].set_visible(False)
    axes_cbar.spines["right"].set_visible(False)
    axes_cbar.spines["bottom"].set_visible(False)
    axes_cbar.spines["left"].set_visible(False)
    axes_cbar.axes.get_xaxis().set_visible(False)
    axes_cbar.axes.get_yaxis().set_visible(False)

    # show EMG
    i_mu = np.where((freq >= r_mu[0]) & (freq <= r_mu[1]))[0]
    # * 1000: to go from mV to uV
    p_mu = np.sqrt(SPEMG[i_mu, :].sum(axis=0) * (freq[1] - freq[0])) #* 1000.0 # back to muV
    axes_emg = plt.axes([0.1, 0.5, 0.8, 0.1], sharex=axes_spec)
    axes_emg.plot(t, p_mu[istart:iend], color='black')
    axes_emg.patch.set_alpha(0.0)
    axes_emg.spines["bottom"].set_visible(False)
    if len(emg_ticks) > 0:
        axes_emg.set_yticks(emg_ticks)
    plt.ylabel('Ampl. ' + '$\mathrm{(\mu V)}$')
    plt.xlim((t[0], t[-1] + 1))
    box_off(axes_emg)

    #if len(fig_file) > 0:
    #    save_figure(fig_file)

    plt.show()


def load_stateidx(ppath, name, ann_name='', newann=True):
    """ load the sleep state file of recording (folder) $ppath/$name
    @Return:
        M,K         sequence of sleep states, sequence of 
                    0'1 and 1's indicating non- and annotated states
    """
    ddir = os.path.join(ppath, name)
    ppath, name = os.path.split(ddir)

    if ann_name == '':
        ann_name = name
    
    #if newann:
    #    sfile = os.path.join(ppath, name, 'remidx_' + ann_name + '.txt')
    #else:
    #    sfile = os.path.join(ppath, name, '2remidx_' + ann_name + '.txt')
    sfile = os.path.join(ppath, name, '3_remidx_' + ann_name + '.txt')

    f = open(sfile, 'r') 
    lines = f.readlines()
    f.close()
    
    n = 0
    for l in lines:
        if re.match('\d', l):
            n += 1
            
    M = np.zeros(n)
    K = np.zeros(n)
    
    i = 0
    for l in lines :
        
        if re.search('^\s+$', l) :
            continue
        if re.search('\s*#', l) :
            continue
        
        if re.match('\d+\s+-?\d+', l) :
            a = re.split('\s+', l)
            M[i] = int(a[0])
            K[i] = int(a[1])
            i += 1
            
    return M,K

def get_sequences(idx, ibreak=1) :  
    """
    get_sequences(idx, ibreak=1)
    idx     -    np.vector of indices
    @RETURN:
    seq     -    list of np.vectors
    """
    diff = idx[1:] - idx[0:-1]
    breaks = np.nonzero(diff>ibreak)[0]
    breaks = np.append(breaks, len(idx)-1)
    
    seq = []    
    iold = 0
    for i in breaks:
        r = list(range(iold, i+1))
        seq.append(idx[r])
        iold = i+1
        
    return seq

def power_spectrum(data, length, dt):
    f, pxx = scipy.signal.welch(data, fs=1/dt, window='hanning', nperseg=int(length),
                                noverlap=int(length/2))
    return pxx, f

def get_snr(ppath, name):
    """
    read and return SR from file $ppath/$name/info.txt 
    """
    fid = open(os.path.join(ppath, name, 'info.txt'), newline=None)
    lines = fid.readlines()
    fid.close()
    values = []
    for l in lines :
        a = re.search("^" + 'SR' + ":" + "\s+(.*)", l)
        if a :
            values.append(a.group(1))            
    return float(values[0])

def get_infoparam(ifile, field):
    """
    NOTE: field is a single string
    and the function does not check for the type
    of the values for field.
    In fact, it just returns the string following field
    """
    fid = open(ifile, newline=None)
    lines = fid.readlines()
    fid.close()
    values = []
    for l in lines :
        a = re.search("^" + field + ":" + "\s+(.*)", l)
        if a :
            values.append(a.group(1))
            
    return values

########################################################################
############# Functions to take raw .mat file and make data frame ######
########################################################################

def find_light(ppath, rec, dark=False):
    """
    Using info file in recording, find light-phase of recording
    """
    act_dur = get_infoparam(os.path.join(ppath, rec, 'info.txt'), 'actual_duration')
    time_param = get_infoparam(os.path.join(ppath, rec, 'info.txt'), 'time')

    start_a, start_b, start_c = [int(i) for i in re.split(':', time_param[0])]    
    dur_a, dur_b, dur_c = [int(i[0:-1]) for i in re.split(':', act_dur[0])]
    
    total_dur = dur_a*3600 + dur_b*60 + dur_c
    start_time = start_a*3600 + start_b*60 + start_c 
    end_time = start_time + total_dur
    
    seven_am = 7*3600
    seven_pm = 19*3600
    seven_am2 = 31*3600
    seven_pm2 = 43*3600
    start = 'light'
    
    M,S = load_stateidx(ppath, rec)
    if end_time < seven_pm:
        nswitch=0
    else:
        remainder = end_time -seven_pm
        nswitch = 1 + int(np.floor(remainder/(12*3600)))
    if dark==False:
        if nswitch==0:
            return [M], nswitch, 0
        elif nswitch==1:
            light = int((seven_pm-start_time)/2.5)
            M1 = M[0:light]
            return [M1], nswitch, 0
        else:
            light = int((seven_pm - start_time)/2.5)
            light2 = int((seven_am2 - start_time)/2.5)+1
            M1 = M[0:light]
            M2 = M[light2:]
            return[M1, M2], nswitch, light2
    else:
        light = int((seven_pm - start_time)/2.5)
        light2 = int((seven_am2 - start_time)/2.5)+1
        Mnight = M[light:light2]
        return [Mnight], 2, light

def split_light(ppath, rec, firsthalf=True):
    act_dur = get_infoparam(os.path.join(ppath, rec, 'info.txt'), 'actual_duration')
    time_param = get_infoparam(os.path.join(ppath, rec, 'info.txt'), 'time')

    start_a, start_b, start_c = [int(i) for i in re.split(':', time_param[0])]    
    dur_a, dur_b, dur_c = [int(i[0:-1]) for i in re.split(':', act_dur[0])]
    
    total_dur = dur_a*3600 + dur_b*60 + dur_c
    start_time = start_a*3600 + start_b*60 + start_c 
    end_time = start_time + total_dur

    seven_am = 7*3600
    seven_pm = 19*3600
    seven_am2 = 31*3600
    seven_pm2 = 43*3600
    start = 'light'

    one_pm = 13*3600

    M,S = load_stateidx(ppath,rec)      

    if end_time < seven_pm:
        nswitch = 0
    else:
        remainder = end_time - seven_pm
        nswitch = 1 + int(np.floor(remainder/(12*3600)))

    if nswitch==0:
        light1 = int((one_pm-start_time)/2.5)
        M1 = M[0:light1]
        M2 = M[light1:]
        if firsthalf:
            return [M1]
        else:
            return [M2]
    elif nswitch==1:
        light1 = int((one_pm-start_time)/2.5)
        light2 = int((seven_pm-start_time)/2.5)
        M1 = M[0:light1]
        M2 = M[light1:light2]
        if firsthalf:
            return [M1]
        else:
            return [M2]
    else:
        light1 = int((one_pm-start_time)/2.5)
        light2 = int((seven_pm-start_time)/2.5)
        light3 = int((seven_am2 - start_time)/2.5)+1
        M1 = M[0:light1]
        M2 = M[light1:light2]
        M3 = M[light3:]
        if firsthalf:
            return [M1,M3]
        else:
            return [M2]

def split_dark(ppath, rec, firsthalf=True):
    act_dur = get_infoparam(os.path.join(ppath, rec, 'info.txt'), 'actual_duration')
    time_param = get_infoparam(os.path.join(ppath, rec, 'info.txt'), 'time')

    start_a, start_b, start_c = [int(i) for i in re.split(':', time_param[0])]    
    dur_a, dur_b, dur_c = [int(i[0:-1]) for i in re.split(':', act_dur[0])]
    
    total_dur = dur_a*3600 + dur_b*60 + dur_c
    start_time = start_a*3600 + start_b*60 + start_c 
    end_time = start_time + total_dur

    seven_am = 7*3600
    seven_pm = 19*3600
    seven_am2 = 31*3600
    seven_pm2 = 43*3600
    start = 'light'

    one_pm = 13*3600
    one_am = 25*3600

    M,S = load_stateidx(ppath, rec) 
    if end_time < seven_pm:
        nswitch = 0
    else:
        remainder = end_time - seven_pm
        nswitch = 1 + int(np.floor(remainder/(12*3600)))

    darkstart = int((seven_pm-start_time)/2.5)
    darkmid = int((one_am-start_time)/2.5)
    darkend = int((seven_am2 - start_time)/2.5)+1

    dark1 = M[darkstart:darkmid]
    dark2 = M[darkmid:darkend]
    if firsthalf:
        return [dark1]
    else:
        return [dark2]


def nts(M):
    """
    Receive list of states coded as number and change to letters representing state
    """
    rvec = [None]*len(M)
    for idx,x in enumerate(M):
        if x==1:
            rvec[idx]='R'
        elif x==2:
            rvec[idx]='W'
        else:
            rvec[idx]='N'
    return rvec

def vecToTup(rvec, start=0):
    """
    Receive list of states and change to a list of tuples (state, duration, starting index)
    """
    result = []
    cnt1 = 0
    i_start = start
    sum1 = 1
    curr = 'a'
    while cnt1 < len(rvec)-1:
        curr = rvec[cnt1]
        nxt = rvec[cnt1+1]
        if curr==nxt:
            sum1+=1
        else:
            result.append((curr, sum1, i_start))
            sum1=1
        cnt1+=1
        i_start+=1
    last = rvec[-1]
    if curr==last:
        result.append((curr, sum1, i_start))
    else:
        result.append((last, sum1, i_start))
    return result

def vecToTup2(rvec, start=0):
    result = []
    cnt1 = 0
    i_start = start
    sum1 = 1
    curr = 'a'
    while cnt1 < len(rvec)-1:
        curr = rvec[cnt1]
        nxt = rvec[cnt1+1]
        if curr==nxt:
            sum1+=1
        else:
            result.append((curr, sum1, i_start))
            sum1=1
        cnt1+=1
        i_start+=1
    result.append((curr, sum1, i_start))
    return result

def ma_thr(tupList, threshold=8):
    """
    Receive list of tuples and changes wake<threshold into MA
    """
    fixtup = []
    fixtup.append(tupList[0])
    for x in tupList[1:]:
        if (x[0]=='W')&(x[1]<=threshold):
            fixtup.append(('MA', x[1], x[2]))
        else:
            fixtup.append(x)
    return fixtup


def nrt_locations(tupList2):
    """
    Receive list of tuples and return index of NREM-->REM transitions
    """
    nrt_locs = []
    cnt1 = 0
    while cnt1 < len(tupList2)-1:
        curr = tupList2[cnt1]
        nxt = tupList2[cnt1+1]
        if (curr[0]=='N')&(nxt[0]=='R'):
            nrt_locs.append(cnt1+1)
        cnt1+=1
    return nrt_locs

def nrt_locations30(tupList2):
    nrt_locs = []
    cnt1 = 0
    while cnt1 < len(tupList2)-1:
        curr = tupList2[cnt1]
        nxt = tupList2[cnt1+1]
        if (curr[0]=='N')&(nxt[0]=='R')&(nxt[1]>=12):
            nrt_locs.append(cnt1+1)
        cnt1+=1
    return nrt_locs

def nrt_locations_cust(tupList2, dur = 12):
    nrt_locs = []
    cnt1 = 0
    while cnt1 < len(tupList2)-1:
        curr = tupList2[cnt1]
        nxt = tupList2[cnt1+1]
        if (curr[0]=='N')&(nxt[0]=='R')&(nxt[1]>=dur):
            nrt_locs.append(cnt1+1)
        cnt1+=1
    return nrt_locs    


def tupToList(tupList, rec):
    """
    Take list of tuples and return multiple lists representing each state
    """
    tupList2 = tupList[1:len(tupList)-1] #truncate first and last elements
    nrt_locs = nrt_locations(tupList2)
    
    r_pre = []
    n_rem = []
    i_wake = []
    mAs = []
    ind_start = []
    ma_cnt = []
    recn = []
    
    cnt2=0
    while cnt2<len(nrt_locs)-1:
        sub = tupList2[nrt_locs[cnt2]:nrt_locs[cnt2+1]]
        nrem=0
        rem=0
        wake=0
        ma=0
        macnt=0
        for y in sub:
            if y[0]=='N':
                nrem+=y[1]
            elif y[0]=='R':
                rem+=y[1]
            elif y[0]=='MA':
                ma+=y[1]
                macnt+=1
            else:
                wake+=y[1]
        r_pre.append(rem)
        n_rem.append(nrem)
        i_wake.append(wake)
        mAs.append(ma)
        ind_start.append(sub[0][2] - sub[0][1] + 1)
        #ind_end.append(sub[len(sub)-1][2])
        ma_cnt.append(macnt)
        recn.append(rec)
        cnt2+=1
    return([r_pre, n_rem, mAs, i_wake, ind_start, recn, ma_cnt])

def lighthalfDF(ppath, recordings, thre=8, firsthalf=True):
    det_list = []
    for rec in recordings:
        print(rec)
        if firsthalf:
            recs = split_light(ppath,rec,True)
        else:
            recs = split_light(ppath,rec,False)
        for idx,x in enumerate(recs):
            Mvec = nts(x)
            Mtup = vecToTup(Mvec,0)
            fix_tup = ma_thr(Mtup, threshold=thre)
            Mlist = tupToList(fix_tup, rec)
            det_list.append(Mlist)
    rems = []
    nrems = []
    mAs = []
    wakes = []
    recn = []
    macnt = []
    for x in det_list:
        rems.extend(x[0])
        nrems.extend(x[1])
        mAs.extend(x[2])
        wakes.extend(x[3])
        recn.extend(x[4])
        macnt.extend(x[5])
    remDF = pd.DataFrame(list(zip(rems, nrems, mAs, wakes, recn, macnt)), 
                         columns = ['rem', 'nrem', 'mA', 'wake', 'recording', 'macnt'])
    remDF['rem'] = remDF['rem'].apply(lambda x: x*2.5)
    remDF['nrem'] = remDF['nrem'].apply(lambda x: x*2.5)
    remDF['mA'] = remDF['mA'].apply(lambda x: x*2.5)
    remDF['wake'] = remDF['wake'].apply(lambda x: x*2.5)
    remDF['inter'] = remDF['nrem'] + remDF['wake'] + remDF['mA']
    remDF['sws'] = remDF['nrem'] + remDF['mA']
    remDF['logsws'] = remDF['sws'].apply(np.log)
    remDF['logsws2'] = remDF['sws'].apply(np.log2)
    remDF['logsws10'] = remDF['sws'].apply(np.log10)

    return remDF

def darkhalfDF(ppath, recordings, thre=8, firsthalf=True):
    det_list = []
    for rec in recordings:
        print(rec)
        recs = split_dark(ppath,rec,firsthalf)
        for idx,x in enumerate(recs):
            Mvec = nts(x)
            Mtup = vecToTup2(Mvec,0)
            fix_tup = ma_thr(Mtup, threshold=thre)
            Mlist = tupToList(fix_tup, rec)
            det_list.append(Mlist)
    rems = []
    nrems = []
    mAs = []
    wakes = []
    recn = []
    macnt = []
    for x in det_list:
        rems.extend(x[0])
        nrems.extend(x[1])
        mAs.extend(x[2])
        wakes.extend(x[3])
        recn.extend(x[4])
        macnt.extend(x[5])
    remDF = pd.DataFrame(list(zip(rems, nrems, mAs, wakes, recn, macnt)), 
                         columns = ['rem', 'nrem', 'mA', 'wake', 'recording', 'macnt'])
    remDF['rem'] = remDF['rem'].apply(lambda x: x*2.5)
    remDF['nrem'] = remDF['nrem'].apply(lambda x: x*2.5)
    remDF['mA'] = remDF['mA'].apply(lambda x: x*2.5)
    remDF['wake'] = remDF['wake'].apply(lambda x: x*2.5)
    remDF['inter'] = remDF['nrem'] + remDF['wake'] + remDF['mA']
    remDF['sws'] = remDF['nrem'] + remDF['mA']
    remDF['logsws'] = remDF['sws'].apply(np.log)
    remDF['logsws2'] = remDF['sws'].apply(np.log2)
    remDF['logsws10'] = remDF['sws'].apply(np.log10)

    return remDF    


def standard_recToDF(ppath, recordings, thre = 8, isDark=False):
    """
    Given ppath and recordings(list), return dataframe containing all info
    """
    det_list = []
    if isDark==False:
        for rec in recordings:
            print(rec)
            recs, nswitch, start = find_light(ppath, rec)
            for idx,x in enumerate(recs):
                Mvec = nts(x)
                Mtup = []
                if idx==0:
                    Mtup = vecToTup(Mvec, start = 0)
                else:
                    Mtup = vecToTup(Mvec, start=start)
                fix_tup = ma_thr(Mtup, threshold=thre)
                Mlist = tupToList(fix_tup, rec)
                det_list.append(Mlist)
    else:
        for rec in recordings:
            print(rec)
            recs, nswitch, start = find_light(ppath, rec, dark=True)
            for idx,x in enumerate(recs):
                Mvec = nts(x)
                Mtup = []
                Mtup = vecToTup2(Mvec, start = start)
                fix_tup = ma_thr(Mtup, threshold=thre)
                Mlist = tupToList(fix_tup, rec)
                det_list.append(Mlist)
    rems = []
    nrems = []
    mAs = []
    wakes = []
    istart = []
    recn = []
    macnt = []
    for x in det_list:
        rems.extend(x[0])
        nrems.extend(x[1])
        mAs.extend(x[2])
        wakes.extend(x[3])
        istart.extend(x[4])
        recn.extend(x[5])
        macnt.extend(x[6])
    remDF = pd.DataFrame(list(zip(rems, nrems, mAs, wakes, istart, recn, macnt)), 
                         columns = ['rem', 'nrem', 'mA', 'wake', 'istart','recording', 'macnt'])
    remDF['rem'] = remDF['rem'].apply(lambda x: x*2.5)
    remDF['nrem'] = remDF['nrem'].apply(lambda x: x*2.5)
    remDF['mA'] = remDF['mA'].apply(lambda x: x*2.5)
    remDF['wake'] = remDF['wake'].apply(lambda x: x*2.5)
    remDF['inter'] = remDF['nrem'] + remDF['wake'] + remDF['mA']
    remDF['sws'] = remDF['nrem'] + remDF['mA']
    remDF['logsws'] = remDF['sws'].apply(np.log)
    remDF['logsws2'] = remDF['sws'].apply(np.log2)
    remDF['logsws10'] = remDF['sws'].apply(np.log10)

    return remDF




################################################################################
######### Functions for Gaussian Mixture Model ##########
################################################################################
def splitData(remDF, byy=30):
    splitCount = 240/byy
    res = []
    cnt1 = 0
    while cnt1 < splitCount:
        subDF = remDF.loc[(remDF.rem>=cnt1*byy)&(remDF.rem<(cnt1+1)*byy)]
        res.append(subDF)
        cnt1+=1
    return res

def gmm_params(splitlogsws):
    khigh = []
    klow = []
    mhigh = []
    mlow = []
    shigh = []
    slow = []
    for idx,data in enumerate(splitlogsws):
        if idx<5:
            gmm = GaussianMixture(n_components=2, tol=0.000001, max_iter=999999999)
            gmm.fit(np.expand_dims(data,1))
            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_.flatten())
            weights = gmm.weights_.flatten()
            k1 = weights[0]
            k2 = weights[1]
            m1 = means[0]
            m2 = means[1]
            s1 = stds[0]
            s2 = stds[1]
            k1 = np.round(k1,6)
            k2 = np.round(k2,6)
            m1 = np.round(m1,6)
            m2 = np.round(m2,6)
            s1 = np.round(s1,6)
            s2 = np.round(s2,6)
            if m1<m2:
                klow.append(k1)
                khigh.append(k2)
                mlow.append(m1)
                mhigh.append(m2)
                slow.append(s1)
                shigh.append(s2)
            else:
                klow.append(k2)
                khigh.append(k1)
                mlow.append(m2)
                mhigh.append(m1)
                slow.append(s2)
                shigh.append(s1)
        else:
            gmm = GaussianMixture(n_components=1, tol=0.000001, max_iter=999999999)
            gmm.fit(np.expand_dims(data,1))
            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_.flatten())
            weights = gmm.weights_.flatten()
            k = weights[0]
            m = means[0]
            s = stds[0]
            k = np.round(k,6)
            m = np.round(m,6)
            s = np.round(s,6)
            khigh.append(k)
            mhigh.append(m)
            shigh.append(s)
            klow.append(None)
            mlow.append(None)
            slow.append(None)
    return khigh,klow,mhigh,mlow,shigh,slow

def dark_gmm(splitlogsws):
    khigh = []
    klow = []
    mhigh = []
    mlow = []
    shigh = []
    slow = []

    for idx,data in enumerate(splitlogsws):
        #if (idx<3)or(idx==5):
        if min(data)<5:
            gmm = GaussianMixture(n_components=2, tol=0.000001, max_iter=999999999)
            gmm.fit(np.expand_dims(data,1))
            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_.flatten())
            weights = gmm.weights_.flatten()
            k1 = weights[0]
            k2 = weights[1]
            m1 = means[0]
            m2 = means[1]
            s1 = stds[0]
            s2 = stds[1]
            k1 = np.round(k1,6)
            k2 = np.round(k2,6)
            m1 = np.round(m1,6)
            m2 = np.round(m2,6)
            s1 = np.round(s1,6)
            s2 = np.round(s2,6)
            if m1<m2:
                klow.append(k1)
                khigh.append(k2)
                mlow.append(m1)
                mhigh.append(m2)
                slow.append(s1)
                shigh.append(s2)
            else:
                klow.append(k2)
                khigh.append(k1)
                mlow.append(m2)
                mhigh.append(m1)
                slow.append(s2)
                shigh.append(s1)
        else:
            gmm = GaussianMixture(n_components=1, tol=0.000001, max_iter=999999999)
            gmm.fit(np.expand_dims(data,1))
            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_.flatten())
            weights = gmm.weights_.flatten()
            k = weights[0]
            m = means[0]
            s = stds[0]
            k = np.round(k,6)
            m = np.round(m,6)
            s = np.round(s,6)
            khigh.append(k)
            mhigh.append(m)
            shigh.append(s)
            klow.append(None)
            mlow.append(None)
            slow.append(None)
        #print(idx)
    return khigh,klow,mhigh,mlow,shigh,slow


###########################################################################
################## Functions for zone comparisons #######################
###########################################################################

def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

def stateIndex(sub, stateInds):
    res = []
    for x in stateInds:
        cur = sub[x]
        curIndex = list(np.arange(cur[2]-cur[1]+1,cur[2]+1))
        res.extend(curIndex)
    return res

def stateSeq(sub, stateInds):
    stateIndices = stateIndex(sub, stateInds)
    stateRanges = ranges(stateIndices)
    stateSeqs = []
    for x in stateRanges:
        stateSeqs.append(np.arange(x[0],x[1]+1))
    return stateSeqs

def isSequential(rem, sws, intersect_x, intersect_y):
    exp_intersect_y = np.exp(intersect_y)
    if rem > max(intersect_x):
        return False
    else:
        curind = np.where(intersect_x==rem)[0][0]
        thresh = exp_intersect_y[curind][0]
        if sws<thresh:
            return True
        else:
            return False


###########################################################################
################# Functions for progression comparisons ###################
###########################################################################
def tupToVec(tupL):
    res = []
    for x in tupL:
        ind1=0
        while ind1<x[1]:
            res.append(x[0])
            ind1+=1
    return res

def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

def athreshold(rempre, threshlist):
    indx = np.arange(7.5, 242.5, 2.5)
    ind = np.where(indx==rempre)
    return threshlist[ind]

def mid_threshold(rempre, threshlist):
    indx = np.arange(7.5, 242.5, 2.5)
    ind = np.where(indx==rempre)
    return threshlist[ind]

def high_threshold(rempre, threshlist):
    indx = np.arange(7.5, 242.5, 2.5)
    ind = np.where(indx==rempre)
    return threshlist[ind]

def findSplitInd(subvec, scaledThresh):
    time_elapsed = 0
    remcnt = 0
    res = -1
    for idx,x in enumerate(subvec):
        if x=='R':
            remcnt+=1
        if (x=='N')or(x=='MA'):
            time_elapsed+=1
        if time_elapsed>=scaledThresh:
            res=idx
            break
    return res-remcnt

def find_refractory(sub, threshold):
    states = np.array([x[0] for x in sub])

    rem = sub[0][1]*2.5
    scaled_refThresh = athreshold(rem, threshold)/2.5

    subFixvec = tupToVec(sub)
    splitInd = findSplitInd(subFixvec, scaled_refThresh)

    if splitInd>0:
        istart = sub[0][2]-sub[0][1]+1
        refstart = istart + int(rem/2.5)
        refend = refstart+splitInd
        permend = sub[-1][2]
        ref_totseq = np.arange(refstart, refend)
        perm_totseq = np.arange(refend, permend+1)

        wakeInds = np.where(states=='W')[0]
        wake_seqs = []
        for x in wakeInds:
            cur = sub[x]
            curseq = list(np.arange(cur[2]-cur[1]+1, cur[2]+1))
            wake_seqs.extend(curseq)

        refseq = [x for x in ref_totseq if x not in wake_seqs]
        permseq = [x for x in perm_totseq if x not in wake_seqs]

        refranges = ranges(refseq)
        permranges = ranges(permseq)

        allRefSeqs = []
        for x in refranges:
            allRefSeqs.append(np.arange(x[0],x[1]+1))
        allPermSeqs = []
        for x in permranges:
            allPermSeqs.append(np.arange(x[0],x[1]+1))

        return allRefSeqs, allPermSeqs
    else:
        return [],[]

def refra_quarters(sub, exp_upper5):
    states = np.array([x[0] for x in sub])

    rem = sub[0][1]*2.5
    scaled_refThresh = athreshold(rem, exp_upper5)/2.5

    subFixvec = tupToVec(sub)
    splitInd = findSplitInd(subFixvec, scaled_refThresh)

    if splitInd>0:
        istart = sub[0][2]-sub[0][1]+1
        refstart = istart + int(rem/2.5)
        refend = refstart + splitInd
        permend = sub[-1][2]

        ref_totseq = np.arange(refstart, refend)

        wakeInds = np.where(states=='W')[0]
        wake_seqs = []
        for x in wakeInds:
            cur = sub[x]
            curseq = list(np.arange(cur[2]-cur[1]+1, cur[2]+1))
            wake_seqs.extend(curseq)

        refseq = [x for x in ref_totseq if x not in wake_seqs]

        if len(refseq)>8:
            refsplitlen = int(len(refseq)/4)
            ref1 = refseq[0:refsplitlen]
            ref2 = refseq[refsplitlen:2*refsplitlen]
            ref3 = refseq[2*refsplitlen:3*refsplitlen]
            ref4 = refseq[3*refsplitlen:]

            ref1ranges = ranges(ref1)
            ref2ranges = ranges(ref2)
            ref3ranges = ranges(ref3)
            ref4ranges = ranges(ref4)

            ref1seq = []
            for x in ref1ranges:
                ref1seq.append(np.arange(x[0],x[1]+1))
            ref2seq = []
            for x in ref2ranges:
                ref2seq.append(np.arange(x[0],x[1]+1))
            ref3seq = []
            for x in ref3ranges:
                ref3seq.append(np.arange(x[0],x[1]+1))
            ref4seq = []
            for x in ref4ranges:
                ref4seq.append(np.arange(x[0],x[1]+1))            
            return ref1seq, ref2seq, ref3seq, ref4seq
        else:
            return [],[],[],[]
    else:
        return [],[],[],[]


def perm_quarters(sub, exp_upper5):
    states = np.array([x[0] for x in sub])

    rem = sub[0][1]*2.5
    scaled_refThresh = athreshold(rem, exp_upper5)/2.5

    subFixvec = tupToVec(sub)
    splitInd = findSplitInd(subFixvec, scaled_refThresh)

    if splitInd>0:
        istart = sub[0][2]-sub[0][1]+1
        refstart = istart + int(rem/2.5)
        refend = refstart + splitInd
        permend = sub[-1][2]

        perm_totseq = np.arange(refend, permend+1)

        wakeInds = np.where(states=='W')[0]
        wake_seqs = []
        for x in wakeInds:
            cur = sub[x]
            curseq = list(np.arange(cur[2]-cur[1]+1, cur[2]+1))
            wake_seqs.extend(curseq)

        permseq = [x for x in perm_totseq if x not in wake_seqs]

        if len(permseq)>8:
            permsplitlen = int(len(permseq)/4)
            perm1 = permseq[0:permsplitlen]
            perm2 = permseq[permsplitlen:2*permsplitlen]
            perm3 = permseq[2*permsplitlen:3*permsplitlen]
            perm4 = permseq[3*permsplitlen:]

            perm1ranges = ranges(perm1)
            perm2ranges = ranges(perm2)
            perm3ranges = ranges(perm3)
            perm4ranges = ranges(perm4)

            perm1seq = []
            for x in perm1ranges:
                perm1seq.append(np.arange(x[0],x[1]+1))
            perm2seq = []
            for x in perm2ranges:
                perm2seq.append(np.arange(x[0],x[1]+1))
            perm3seq = []
            for x in perm3ranges:
                perm3seq.append(np.arange(x[0],x[1]+1))
            perm4seq = []
            for x in perm4ranges:
                perm4seq.append(np.arange(x[0],x[1]+1))            
            return perm1seq, perm2seq, perm3seq, perm4seq
        else:
            return [],[],[],[]
    else:
        return [],[],[],[]


def perm_quarters_exclude40(sub, exp_upper5):
    states = np.array([x[0] for x in sub])

    rem = sub[0][1]*2.5
    scaled_refThresh = athreshold(rem, exp_upper5)/2.5

    subFixvec = tupToVec(sub)
    splitInd = findSplitInd(subFixvec, scaled_refThresh)

    if splitInd>0:
        istart = sub[0][2]-sub[0][1]+1
        refstart = istart + int(rem/2.5)
        refend = refstart + splitInd
        permend = sub[-1][2]

        perm_totseq = np.arange(refend, permend+1)

        wakeInds = np.where(states=='W')[0]
        wake_seqs = []
        for x in wakeInds:
            cur = sub[x]
            curseq = list(np.arange(cur[2]-cur[1]+1, cur[2]+1))
            wake_seqs.extend(curseq)

        permseq = [x for x in perm_totseq if x not in wake_seqs]
        permseq = permseq[:len(permseq)-16]

        if len(permseq)>8:
            permsplitlen = int(len(permseq)/4)
            perm1 = permseq[0:permsplitlen]
            perm2 = permseq[permsplitlen:2*permsplitlen]
            perm3 = permseq[2*permsplitlen:3*permsplitlen]
            perm4 = permseq[3*permsplitlen:]

            perm1ranges = ranges(perm1)
            perm2ranges = ranges(perm2)
            perm3ranges = ranges(perm3)
            perm4ranges = ranges(perm4)

            perm1seq = []
            for x in perm1ranges:
                perm1seq.append(np.arange(x[0],x[1]+1))
            perm2seq = []
            for x in perm2ranges:
                perm2seq.append(np.arange(x[0],x[1]+1))
            perm3seq = []
            for x in perm3ranges:
                perm3seq.append(np.arange(x[0],x[1]+1))
            perm4seq = []
            for x in perm4ranges:
                perm4seq.append(np.arange(x[0],x[1]+1))            
            return perm1seq, perm2seq, perm3seq, perm4seq
        else:
            return [],[],[],[]
    else:
        return [],[],[],[]


def splitbyeight(sub):
    rem = sub[0][1]*2.5
    start = sub[1][2]-sub[1][1]+1
    end = sub[-1][2]

    totseq = np.arange(start, end+1)
    splitlen = int(len(totseq)/8)

    if splitlen>=1:
        seq1 = np.array(totseq[0:splitlen])
        seq2 = np.array(totseq[splitlen:2*splitlen])
        seq3 = np.array(totseq[2*splitlen:3*splitlen])
        seq4 = np.array(totseq[3*splitlen:4*splitlen])
        seq5 = np.array(totseq[4*splitlen:5*splitlen])
        seq6 = np.array(totseq[5*splitlen:6*splitlen])
        seq7 = np.array(totseq[6*splitlen:7*splitlen])
        seq8 = np.array(totseq[7*splitlen:])
        return [seq1,seq2,seq3,seq4,seq5,seq6,seq7,seq8]
    else:
        return None

def splitbytwelve(sub):
    rem = sub[0][1]*2.5
    start = sub[1][2]-sub[1][1]+1
    end = sub[-1][2]

    totseq = np.arange(start, end+1)
    splitlen = int(len(totseq)/12)  

    if splitlen>=1:
        seq1 = np.array(totseq[0:splitlen])
        seq2 = np.array(totseq[splitlen:2*splitlen])
        seq3 = np.array(totseq[2*splitlen:3*splitlen])
        seq4 = np.array(totseq[3*splitlen:4*splitlen])
        seq5 = np.array(totseq[4*splitlen:5*splitlen])
        seq6 = np.array(totseq[5*splitlen:6*splitlen])
        seq7 = np.array(totseq[6*splitlen:7*splitlen])      
        seq8 = np.array(totseq[7*splitlen:8*splitlen])
        seq9 = np.array(totseq[8*splitlen:9*splitlen])
        seq10 = np.array(totseq[9*splitlen:10*splitlen])
        seq11 = np.array(totseq[10*splitlen:11*splitlen])
        seq12 = np.array(totseq[11*splitlen:])
        return [seq1,seq2,seq3,seq4,seq5,seq6,seq7,seq8,seq9,seq10,seq11,seq12]
    else:
        return None

def all_lowmid(sub, mid1):
    states = np.array([x[0] for x in sub])

    rem = sub[0][1]*2.5
    inter = 0
    for x in sub[1:]:
        inter+=x[1]*2.5

    midthresh = mid_threshold(rem,mid1)

    if inter>midthresh:
        scaledmidthresh = int(midthresh/2.5)

        lowstart = sub[1][2]-sub[1][1]+1
        lowend = lowstart + scaledmidthresh
        midend = sub[-1][2]

        lowseq = np.arange(lowstart, lowend+1)
        midseq = np.arange(lowend+1, midend+1)

        return [lowseq], [midseq]
    else:
        return [],[]




def spindlema_lowmid_quarters(sub, mid1):
    rem = sub[0][1]*2.5
    midthresh = mid_threshold(rem, mid1)
    scaledmidthresh = int(midthresh/2.5)

    lowstart = sub[1][2]-sub[1][1]+1
    lowend = lowstart + scaledmidthresh
    midend = sub[-1][2]

    if (midend>=lowend+4)&(scaledmidthresh>=4):
        lowseq = np.arange(lowstart, lowend+1)
        midseq = np.arange(lowend+1, midend+1)

        lowsplitlen = int(len(lowseq)/4)
        lows1 = lowseq[0:lowsplitlen]
        lows2 = lowseq[lowsplitlen:2*lowsplitlen]
        lows3 = lowseq[2*lowsplitlen:3*lowsplitlen]
        lows4 = lowseq[3*lowsplitlen:]

        midsplitlen = int(len(midseq)/4)
        mids1 = midseq[0:midsplitlen]
        mids2 = midseq[midsplitlen:2*midsplitlen]
        mids3 = midseq[2*midsplitlen:3*midsplitlen]
        mids4 = midseq[3*midsplitlen:]

        low1range = ((lows1[0], lows1[-1]))
        low2range = ((lows2[0], lows2[-1]))
        low3range = ((lows3[0], lows3[-1]))
        low4range = ((lows4[0], lows4[-1]))

        mid1range = ((mids1[0], mids1[-1]))
        mid2range = ((mids2[0], mids2[-1]))
        mid3range = ((mids3[0], mids3[-1]))
        mid4range = ((mids4[0], mids4[-1]))

        return [low1range,low2range,low3range,low4range,mid1range,mid2range,mid3range,mid4range]
    else:
        None

def lowmid_quarters_sws(sub, mid1, fixvec):
    rem = sub[0][1]*2.5
    midthresh = mid_threshold(rem, mid1)
    scaledmidthresh = int(midthresh/2.5)

    lowstart = sub[1][2]-sub[1][1]+1
    lowend = lowstart + scaledmidthresh
    midend = sub[-1][2]

    if (midend>=lowend+4)&(scaledmidthresh>=4):
        lowseq = np.arange(lowstart, lowend+1)
        midseq = np.arange(lowend+1, midend+1)

        lowsplitlen = int(len(lowseq)/4)
        lows1 = lowseq[0:lowsplitlen]
        lows2 = lowseq[lowsplitlen:2*lowsplitlen]
        lows3 = lowseq[2*lowsplitlen:3*lowsplitlen]
        lows4 = lowseq[3*lowsplitlen:]

        midsplitlen = int(len(midseq)/4)
        mids1 = midseq[0:midsplitlen]
        mids2 = midseq[midsplitlen:2*midsplitlen]
        mids3 = midseq[2*midsplitlen:3*midsplitlen]
        mids4 = midseq[3*midsplitlen:]

        low1seq = [np.arange(lows1[0],lows1[-1]+1)][0]
        low2seq = [np.arange(lows2[0],lows2[-1]+1)][0]
        low3seq = [np.arange(lows3[0],lows3[-1]+1)][0]
        low4seq = [np.arange(lows4[0],lows4[-1]+1)][0]

        mid1seq = [np.arange(mids1[0],mids1[-1]+1)][0]
        mid2seq = [np.arange(mids2[0],mids2[-1]+1)][0]
        mid3seq = [np.arange(mids3[0],mids3[-1]+1)][0]
        mid4seq = [np.arange(mids4[0],mids4[-1]+1)][0]

        low1vec = np.array(fixvec[lows1[0]:lows1[-1]+1])
        low2vec = np.array(fixvec[lows2[0]:lows2[-1]+1])
        low3vec = np.array(fixvec[lows3[0]:lows3[-1]+1])
        low4vec = np.array(fixvec[lows4[0]:lows4[-1]+1])

        mid1vec = np.array(fixvec[mids1[0]:mids1[-1]+1])
        mid2vec = np.array(fixvec[mids2[0]:mids2[-1]+1])
        mid3vec = np.array(fixvec[mids3[0]:mids3[-1]+1])
        mid4vec = np.array(fixvec[mids4[0]:mids4[-1]+1])

        low1_nwakeinds = np.where(low1vec!='W')[0]
        low2_nwakeinds = np.where(low2vec!='W')[0]
        low3_nwakeinds = np.where(low3vec!='W')[0]
        low4_nwakeinds = np.where(low4vec!='W')[0]

        mid1_nwakeinds = np.where(mid1vec!='W')[0]
        mid2_nwakeinds = np.where(mid2vec!='W')[0]
        mid3_nwakeinds = np.where(mid3vec!='W')[0]
        mid4_nwakeinds = np.where(mid4vec!='W')[0]

        low1array = low1seq[low1_nwakeinds]
        low2array = low2seq[low2_nwakeinds]
        low3array = low3seq[low3_nwakeinds]
        low4array = low4seq[low4_nwakeinds]

        mid1array = mid1seq[mid1_nwakeinds]
        mid2array = mid2seq[mid2_nwakeinds]
        mid3array = mid3seq[mid3_nwakeinds]
        mid4array = mid4seq[mid4_nwakeinds]

        low1ranges = ranges(low1array)
        low2ranges = ranges(low2array)
        low3ranges = ranges(low3array)
        low4ranges = ranges(low4array)

        mid1ranges = ranges(mid1array)
        mid2ranges = ranges(mid2array)
        mid3ranges = ranges(mid3array)
        mid4ranges = ranges(mid4array)

        low1_swsseq = []
        low2_swsseq = []
        low3_swsseq = []
        low4_swsseq = []
        mid1_swsseq = []
        mid2_swsseq = []
        mid3_swsseq = []
        mid4_swsseq = []

        for x in low1ranges:
            start = x[0]
            end = x[1]+1
            low1_swsseq.append(np.arange(start,end))
        for x in low2ranges:
            start = x[0]
            end = x[1]+1
            low2_swsseq.append(np.arange(start,end))
        for x in low3ranges:
            start = x[0]
            end = x[1]+1
            low3_swsseq.append(np.arange(start,end))
        for x in low4ranges:
            start = x[0]
            end = x[1]+1
            low4_swsseq.append(np.arange(start,end))

        for x in mid1ranges:
            start = x[0]
            end = x[1]+1    
            mid1_swsseq.append(np.arange(start,end))
        for x in mid2ranges:
            start = x[0]
            end = x[1]+1    
            mid2_swsseq.append(np.arange(start,end))
        for x in mid3ranges:
            start = x[0]
            end = x[1]+1    
            mid3_swsseq.append(np.arange(start,end))
        for x in mid4ranges:
            start = x[0]
            end = x[1]+1    
            mid4_swsseq.append(np.arange(start,end))
        return low1_swsseq,low2_swsseq,low3_swsseq,low4_swsseq,mid1_swsseq,mid2_swsseq,mid3_swsseq,mid4_swsseq
    else:
        return [],[],[],[],[],[],[],[]


def spindle_lowmid_quarters(sub, mid1):
    rem = sub[0][1]*2.5
    midthresh = mid_threshold(rem, mid1)
    scaledmidthresh = int(midthresh/2.5)

    lowstart = sub[1][2]-sub[1][1]+1
    lowend = lowstart + scaledmidthresh
    midend = sub[-1][2]

    if (midend>=lowend+4)&(scaledmidthresh>=4):
        lowseq = np.arange(lowstart, lowend+1)
        midseq = np.arange(lowend+1, midend+1)

        lowsplitlen = int(len(lowseq)/4)
        lows1 = lowseq[0:lowsplitlen]
        lows2 = lowseq[lowsplitlen:2*lowsplitlen]
        lows3 = lowseq[2*lowsplitlen:3*lowsplitlen]
        lows4 = lowseq[3*lowsplitlen:]

        midsplitlen = int(len(midseq)/4)
        mids1 = midseq[0:midsplitlen]
        mids2 = midseq[midsplitlen:2*midsplitlen]
        mids3 = midseq[2*midsplitlen:3*midsplitlen]
        mids4 = midseq[3*midsplitlen:]

        low1ranges = ranges(lows1)[0]
        low2ranges = ranges(lows2)[0]
        low3ranges = ranges(lows3)[0]
        low4ranges = ranges(lows4)[0]

        mid1ranges = ranges(mids1)[0]
        mid2ranges = ranges(mids2)[0]
        mid3ranges = ranges(mids3)[0]
        mid4ranges = ranges(mids4)[0]

        return low1ranges,low2ranges,low3ranges,low4ranges,mid1ranges,mid2ranges,mid3ranges,mid4ranges
    else:
        return [],[],[],[],[],[],[],[]

def spindle_lowmid_quarters_sws(sub, mid1, fixvec):
    rem = sub[0][1]*2.5
    midthresh = mid_threshold(rem, mid1)
    scaledmidthresh = int(midthresh/2.5)

    lowstart = sub[1][2]-sub[1][1]+1
    lowend = lowstart + scaledmidthresh
    midend = sub[-1][2]

    if (midend>=lowend+4)&(scaledmidthresh>=4):
        lowseq = np.arange(lowstart, lowend+1)
        midseq = np.arange(lowend+1, midend+1)

        lowsplitlen = int(len(lowseq)/4)
        lows1 = lowseq[0:lowsplitlen]
        lows2 = lowseq[lowsplitlen:2*lowsplitlen]
        lows3 = lowseq[2*lowsplitlen:3*lowsplitlen]
        lows4 = lowseq[3*lowsplitlen:]

        midsplitlen = int(len(midseq)/4)
        mids1 = midseq[0:midsplitlen]
        mids2 = midseq[midsplitlen:2*midsplitlen]
        mids3 = midseq[2*midsplitlen:3*midsplitlen]
        mids4 = midseq[3*midsplitlen:]

        low1seq = [np.arange(lows1[0],lows1[-1]+1)][0]
        low2seq = [np.arange(lows2[0],lows2[-1]+1)][0]
        low3seq = [np.arange(lows3[0],lows3[-1]+1)][0]
        low4seq = [np.arange(lows4[0],lows4[-1]+1)][0]

        mid1seq = [np.arange(mids1[0],mids1[-1]+1)][0]
        mid2seq = [np.arange(mids2[0],mids2[-1]+1)][0]
        mid3seq = [np.arange(mids3[0],mids3[-1]+1)][0]
        mid4seq = [np.arange(mids4[0],mids4[-1]+1)][0]

        low1vec = np.array(fixvec[lows1[0]:lows1[-1]+1])
        low2vec = np.array(fixvec[lows2[0]:lows2[-1]+1])
        low3vec = np.array(fixvec[lows3[0]:lows3[-1]+1])
        low4vec = np.array(fixvec[lows4[0]:lows4[-1]+1])

        mid1vec = np.array(fixvec[mids1[0]:mids1[-1]+1])
        mid2vec = np.array(fixvec[mids2[0]:mids2[-1]+1])
        mid3vec = np.array(fixvec[mids3[0]:mids3[-1]+1])
        mid4vec = np.array(fixvec[mids4[0]:mids4[-1]+1])

        low1_nwakeinds = np.where(low1vec!='W')[0]
        low2_nwakeinds = np.where(low2vec!='W')[0]
        low3_nwakeinds = np.where(low3vec!='W')[0]
        low4_nwakeinds = np.where(low4vec!='W')[0]

        mid1_nwakeinds = np.where(mid1vec!='W')[0]
        mid2_nwakeinds = np.where(mid2vec!='W')[0]
        mid3_nwakeinds = np.where(mid3vec!='W')[0]
        mid4_nwakeinds = np.where(mid4vec!='W')[0]

        low1ranges = ranges(lows1)[0]
        low2ranges = ranges(lows2)[0]
        low3ranges = ranges(lows3)[0]
        low4ranges = ranges(lows4)[0]
        mid1ranges = ranges(mids1)[0]
        mid2ranges = ranges(mids2)[0]
        mid3ranges = ranges(mids3)[0]
        mid4ranges = ranges(mids4)[0]

        return low1ranges,len(low1_nwakeinds),low2ranges,len(low2_nwakeinds),low3ranges,len(low3_nwakeinds),low4ranges,len(low4_nwakeinds),mid1ranges,len(mid1_nwakeinds),mid2ranges,len(mid2_nwakeinds),mid3ranges,len(mid3_nwakeinds),mid4ranges,len(mid4_nwakeinds)
    else:
        return [],0,[],0,[],0,[],0,[],0,[],0,[],0,[],0

def ma_lowmid_quarters(sub, mid1):
    rem = sub[0][1]*2.5
    midthresh = mid_threshold(rem, mid1)
    scaledmidthresh = int(midthresh/2.5)

    lowstart = sub[1][2]-sub[1][1]+1
    lowend = lowstart + scaledmidthresh
    midend = sub[-1][2]

    if (midend>=lowend+4)&(scaledmidthresh>=4):
        inter = 0
        for x in sub[1:]:
            inter+=x[1]*2.5

        subsub = sub[1:]
        subFixvec = tupToVec(subsub)

        lowvec = subFixvec[0:scaledmidthresh]
        midvec = subFixvec[scaledmidthresh:]

        lowsplitlen = int(len(lowvec)/4)
        midsplitlen = int(len(midvec)/4)

        lows1 = lowvec[0:lowsplitlen]
        lows2 = lowvec[lowsplitlen:2*lowsplitlen]
        lows3 = lowvec[2*lowsplitlen:3*lowsplitlen]
        lows4 = lowvec[3*lowsplitlen:]

        mids1 = midvec[0:midsplitlen]
        mids2 = midvec[midsplitlen:2*midsplitlen]
        mids3 = midvec[2*midsplitlen:3*midsplitlen]
        mids4 = midvec[3*midsplitlen:]

        lowtup1 = vecToTup(lows1)
        lowtup2 = vecToTup(lows2)
        lowtup3 = vecToTup(lows3)
        lowtup4 = vecToTup(lows4)

        midtup1 = vecToTup(mids1)
        midtup2 = vecToTup(mids2)
        midtup3 = vecToTup(mids3)
        midtup4 = vecToTup(mids4)

        macnt1=0
        macnt2=0
        macnt3=0
        macnt4=0
        macnt5=0
        macnt6=0
        macnt7=0
        macnt8=0

        for x in lowtup1:
            if x[0]=='MA':
                macnt1+=1
        for x in lowtup2:
            if x[0]=='MA':
                macnt2+=1
        for x in lowtup3:
            if x[0]=='MA':
                macnt3+=1
        for x in lowtup4:
            if x[0]=='MA':
                macnt4+=1
        for x in midtup1:
            if x[0]=='MA':
                macnt5+=1
        for x in midtup2:
            if x[0]=='MA':
                macnt6+=1
        for x in midtup3:
            if x[0]=='MA':
                macnt7+=1
        for x in midtup4:
            if x[0]=='MA':
                macnt8+=1
        low1last = lowtup1[-1]
        low2first = lowtup2[0]
        low2last = lowtup2[-1]
        low3first = lowtup3[0]
        low3last = lowtup3[-1]
        low4first = lowtup4[0]
        low4last = lowtup4[-1]
        mid1first = midtup1[0]
        mid1last = midtup1[-1]
        mid2first = midtup2[0]
        mid2last = midtup2[-1]
        mid3first = midtup3[0]
        mid3last = midtup3[-1]
        mid4first = midtup4[0]
        mid4last = midtup4[-1]
        if (low1last[0]=='MA')&(low2first[0]=='MA'):
            if low1last[1]<=low2first[1]:
                macnt1-=1
            else:
                macnt2-=1
        if (low2last[0]=='MA')&(low3first[0]=='MA'):
            if low2last[1]<=low3first[1]:
                macnt2-=1
            else:
                macnt3-=1
        if (low3last[0]=='MA')&(low4first[0]=='MA'):
            if low3last[1]<=low4first[1]:
                macnt3-=1
            else:
                macnt4-=1
        if (low4last[0]=='MA')&(mid1first[0]=='MA'):
            if low4last[1]<=mid1first[1]:
                macnt4-=1
            else:
                macnt5-=1
        if (mid1last[0]=='MA')&(mid2first[0]=='MA'):
            if mid1last[1]<=mid2first[1]:
                macnt5-=1
            else:
                macnt6-=1
        if (mid2last[0]=='MA')&(mid3first[0]=='MA'):
            if mid2last[1]<=mid3first[1]:
                macnt6-=1
            else:
                macnt7-=1
        if (mid3last[0]=='MA')&(mid4first[0]=='MA'):
            if mid3last[1]<=mid4first[1]:
                macnt7-=1
            else:
                macnt8-=1
        lowlen = midthresh
        midlen = inter-midthresh
        marate1 = macnt1/float((lowlen/4))*60
        marate2 = macnt2/float((lowlen/4))*60
        marate3 = macnt3/float((lowlen/4))*60
        marate4 = macnt4/float((lowlen/4))*60
        marate5 = macnt5/float((midlen/4))*60
        marate6 = macnt6/float((midlen/4))*60
        marate7 = macnt7/float((midlen/4))*60
        marate8 = macnt8/float((midlen/4))*60
        return marate1,marate2,marate3,marate4,marate5,marate6,marate7,marate8
    else:
        return None,None,None,None,None,None,None,None


###########################################################################
################# Functions for absolute progression comparisons ##########
###########################################################################

def mid_split_absolute(sub, mid1): #for spectral density
    rem = sub[0][1]*2.5
    midthresh = mid_threshold(rem, mid1)
    scaledmidthresh = int(midthresh/2.5)

    lowstart = sub[1][2]-sub[1][1]+1
    lowend = lowstart + scaledmidthresh
    midend = sub[-1][2]

    totseq = np.arange(lowstart, midend+1)
    subseqs = []

    for i in np.arange(0,800,4):
        first = i
        last = i+4
        if last<len(totseq):
            sub = totseq[first:last]
        else:
            sub = np.array([])
        subseqs.append(sub)
    return subseqs


def mid_split_absolute3(sub, mid1): #for spectral density
    rem = sub[0][1]*2.5
    midthresh = mid_threshold(rem, mid1)
    scaledmidthresh = int(midthresh/2.5)

    lowstart = sub[1][2]-sub[1][1]+1
    lowend = lowstart + scaledmidthresh
    midend = sub[-1][2]

    totseq = np.arange(lowstart, midend+1)
    subseqs = []

    for i in np.arange(0,800,24):
        first = i
        last = i+24
        if last<len(totseq):
            sub = totseq[first:last]
        else:
            sub = np.array([])
        subseqs.append(sub)
    return subseqs  

def high_split_absolute3(sub, mid1): #for spectral density
    rem = sub[0][1]*2.5
    midthresh = mid_threshold(rem, mid1)
    scaledmidthresh = int(midthresh/2.5)

    lowstart = sub[1][2]-sub[1][1]+1
    lowend = lowstart + scaledmidthresh
    midend = sub[-1][2]

    totseq = np.arange(lowstart, midend+1)
    subseqs = []

    for i in np.arange(0,3320,24):
        first = i
        last = i+24
        if last<len(totseq):
            sub = totseq[first:last]
        else:
            sub = np.array([])
        subseqs.append(sub)
    return subseqs  

def mid_split_absolute60(sub, mid1): #for spindles and mas
    rem = sub[0][1]*2.5
    midthresh = mid_threshold(rem, mid1)
    scaledmidthresh = int(midthresh/2.5)

    lowstart = sub[1][2]-sub[1][1]+1
    lowend = lowstart + scaledmidthresh
    midend = sub[-1][2]

    totseq = np.arange(lowstart, midend+1)
    seqlims = []

    for i in np.arange(0,800,24):
        first = i
        last = i+24
        if last<len(totseq):
            subseq = totseq[first:last]
            subrange = ranges(subseq)[0]
            seqlims.append(subrange)
        else:
            seqlims.append((-1,-1))
    return seqlims

def spin_high_split_absolute60(sub, high1): #for spindles and mas
    rem = sub[0][1]*2.5

    lowstart = sub[1][2]-sub[1][1]+1
    highend = sub[-1][2]

    totseq = np.arange(lowstart, highend+1)
    seqlims = []

    for i in np.arange(0,3320,24):
        first = i
        last = i+24
        if last<len(totseq):
            subseq = totseq[first:last]
            subrange = ranges(subseq)[0]
            seqlims.append(subrange)
        else:
            seqlims.append((-1,-1))
    return seqlims


def sixty_sec_splits(sub, thre1):
    ranges = []
    rem = sub[0][1]*2.5

    lowstart = sub[1][2]-sub[1][1]+1
    midend = sub[-1][2]

    totseq = np.arange(lowstart, midend+1)  
    numdiv = int(len(totseq)/24)
    if numdiv==0:
        ranges=[]
    else:
        for i in range(numdiv):
            seqstart = 24*i
            seqend = 24*(i+1)
            subseq = totseq[seqstart:seqend]
            ranges.append((subseq[0],subseq[-1]))
    return ranges




def high_split_absolute(sub, high1):
    rem = sub[0][1]*2.5

    lowstart = sub[1][2] - sub[1][1]+1
    highend = sub[-1][2]

    totseq = np.arange(lowstart, highend+1)
    subseqs = []

    for i in np.arange(0,3320, 4):
        first = i
        last = i+4
        if last<len(totseq):
            sub = totseq[first:last]
        else:
            sub = np.array([])
        subseqs.append(sub)
    return subseqs

def high_split_absolute50(sub, high1):
    rem = sub[0][1]*2.5

    lowstart = sub[1][2] - sub[1][1]+1
    highend = sub[-1][2]

    totseq = np.arange(lowstart, highend+1)
    subseqs = []

    for i in np.arange(0,3320, 20):
        first = i
        last = i+4
        if last<len(totseq):
            sub = totseq[first:last]
        else:
            sub = np.array([])
        subseqs.append(sub)
    return subseqs

def high_split_absolute60(sub, high1):
    rem = sub[0][1]*2.5

    lowstart = sub[1][2] - sub[1][1]+1
    highend = sub[-1][2]

    totseq = np.arange(lowstart, highend+1)
    subseqs = []

    for i in np.arange(0,3320, 24):
        first = i
        last = i+24
        if last<len(totseq):
            sub = totseq[first:last]
        else:
            sub = np.array([])
        subseqs.append(sub)
    return subseqs  

def high_split_absolute100(sub, high1):
    rem = sub[0][1]*2.5

    lowstart = sub[1][2] - sub[1][1]+1
    highend = sub[-1][2]

    totseq = np.arange(lowstart, highend+1)
    subseqs = []

    for i in np.arange(0,3340, 40):
        first = i
        last = i+4
        if last<len(totseq):
            sub = totseq[first:last]
        else:
            sub = np.array([])
        subseqs.append(sub)
    return subseqs      

def thirty_sec_splits(sub, mid1):
    ranges = []

    rem = sub[0][1]*2.5
    midthresh = mid_threshold(rem, mid1)
    scaledmidthresh = int(midthresh/2.5)

    lowstart = sub[1][2]-sub[1][1]+1
    lowend = lowstart + scaledmidthresh
    midend = sub[-1][2]

    totseq = np.arange(lowstart, midend+1)  
    numdiv = int(len(totseq)/12)
    if numdiv==0:
        ranges = []
    else:
        for i in range(numdiv):
            seqstart = 12*i
            seqend = 12*(i+1)
            subseq = totseq[seqstart:seqend]
            ranges.append((subseq[0],subseq[-1]))
    return ranges



def high_sixty_sec_splits(sub):
    ranges = []
    rem = sub[0][1]*2.5

    lowstart = sub[1][2]-sub[1][1]+1
    midend = sub[-1][2]

    totseq = np.arange(lowstart, midend+1)  
    numdiv = int(len(totseq)/24)
    if numdiv==0:
        ranges=[]
    else:
        for i in range(numdiv):
            seqstart = 24*i
            seqend = 24*(i+1)
            subseq = totseq[seqstart:seqend]
            ranges.append((subseq[0],subseq[-1]))
    return ranges

########################################################################################

def wake_tensecsplits(sub):
    allSeqs = []
    start = sub[1][2]-sub[1][1]+1
    end = sub[-1][2]
    totseq = np.arange(start, end+1)

    for i in range(400):
        seqstart = 4*i
        seqend = 4*(i+1)
        if seqend>=len(totseq):
            subseq=[]
        else:
            subseq = totseq[seqstart:seqend]
        allSeqs.append(subseq)

    return allSeqs


def wake_sixtysecsplits(sub):
    allSeqs = []
    start = sub[1][2]-sub[1][1]+1
    end = sub[-1][2]
    totseq = np.arange(start, end+1)

    for i in range(100):
        seqstart = 24*i
        seqend = 24*(i+1)
        if seqend>=len(totseq):
            subseq=[]
        else:
            subseq = totseq[seqstart:seqend]
        allSeqs.append(subseq)

    return allSeqs  

#######################################################################################
def post_tupToList(tupList, rec):
    tupList2 = tupList[1:len(tupList)-1] #truncate first and last elements
   
    nrt_locs = nrt_locations(tupList2)
    
    r_pre = []
    n_rem = []
    i_wake = []
    mAs = []
    ind_start = []
    ma_cnt = []
    recn = []
    r_post = []
    
    cnt2=0
    while cnt2<len(nrt_locs)-2:
        sub = tupList2[nrt_locs[cnt2]:nrt_locs[cnt2+1]]
        sub2 = tupList2[nrt_locs[cnt2+1]:nrt_locs[cnt2+2]]
        nrem=0
        rem=0
        wake=0
        ma=0
        macnt=0
        rempost=sub2[0][1]
        for y in sub:
            if y[0]=='N':
                nrem+=y[1]
            elif y[0]=='R':
                rem+=y[1]
            elif y[0]=='MA':
                ma+=y[1]
                macnt+=1
            else:
                wake+=y[1]
        r_pre.append(rem)
        n_rem.append(nrem)
        i_wake.append(wake)
        mAs.append(ma)
        ind_start.append(sub[0][2] - sub[0][1] + 1)
        #ind_end.append(sub[len(sub)-1][2])
        ma_cnt.append(macnt)
        r_post.append(rempost)
        recn.append(rec)
        cnt2+=1
    return([r_pre, n_rem, mAs, i_wake, ind_start, recn, ma_cnt, r_post])


def post_recToDF(ppath, recordings, thre = 8, isDark=False):
    det_list = []
    if isDark==False:
        for rec in recordings:
            print(rec)
            recs, nswitch, start = find_light(ppath, rec)
            for idx,x in enumerate(recs):
                Mvec = nts(x)
                Mtup = []
                if idx==0:
                    Mtup = vecToTup(Mvec, start = 0)
                else:
                    Mtup = vecToTup(Mvec, start=start)
                fix_tup = ma_thr(Mtup, threshold=thre)
                Mlist = post_tupToList(fix_tup, rec)
                det_list.append(Mlist)
    else:
        for rec in recordings:
            print(rec)
            recs, nswitch, start = find_light(ppath, rec, dark=True)
            for idx,x in enumerate(recs):
                Mvec = nts(x)
                Mtup = []
                Mtup = vecToTup(Mvec, start = start)
                fix_tup = ma_thr(Mtup, threshold=thre)
                Mlist = post_tupToList(fix_tup, rec)
                det_list.append(Mlist)
    rems = []
    nrems = []
    mAs = []
    wakes = []
    istart = []
    recn = []
    macnt = []
    rpost = []
    for x in det_list:
        rems.extend(x[0])
        nrems.extend(x[1])
        mAs.extend(x[2])
        wakes.extend(x[3])
        istart.extend(x[4])
        recn.extend(x[5])
        macnt.extend(x[6])
        rpost.extend(x[7])
    remDF = pd.DataFrame(list(zip(rems, nrems, mAs, wakes, istart, recn, macnt, rpost)), 
                         columns = ['rem', 'nrem', 'mA', 'wake', 'istart','recording', 'macnt', 'rpost'])
    remDF['rem'] = remDF['rem'].apply(lambda x: x*2.5)
    remDF['nrem'] = remDF['nrem'].apply(lambda x: x*2.5)
    remDF['mA'] = remDF['mA'].apply(lambda x: x*2.5)
    remDF['wake'] = remDF['wake'].apply(lambda x: x*2.5)
    remDF['rpost'] = remDF['rpost'].apply(lambda x: x*2.5)
    remDF['inter'] = remDF['nrem'] + remDF['wake'] + remDF['mA']
    remDF['sws'] = remDF['nrem'] + remDF['mA']
    remDF['logsws'] = remDF['sws'].apply(np.log)

    return remDF