data_path = './UMMC_DATA/';
subject_number = 4004;
Fs = 128; % Simband sampling frequency


load(data_path + string(subject_number) + '.mat');
load(data_path + string(subject_number) + '_ground_truth.mat');

% 8-channel ppg from Samsung Simband
ppg = data.physiosignal.ppg;
ecg = data.physiosignal.ecg;

% convert unix timestamps to datetime
% time = datetime(data.unixTimeStamps/1000, 'ConvertFrom', 'epochtime');

ppg_signal = [];

ppg_signal = [ppg_signal; ppg.a.signal];
% ppg_signal = [ppg_signal; ppg.b.signal];
% ppg_signal = [ppg_signal; ppg.c.signal];
% ppg_signal = [ppg_signal; ppg.d.signal];
% ppg_signal = [ppg_signal; ppg.e.signal];
% ppg_signal = [ppg_signal; ppg.f.signal];
% ppg_signal = [ppg_signal; ppg.g.signal];
% ppg_signal = [ppg_signal; ppg.h.signal];

tseries = timeseries(ppg_signal, data.unixTimeStamps);
tseries.Name = 'PPG Time Series';

% ts_a = timeseries()
% 
% tt = table();
% tt.Timestamp = transpose(data.unixTimeStamps);
% tt.PPG_A = transpose(ppg.a.signal);
% tt.PPG_B = transpose(ppg.b.signal);
% tt.PPG_C = transpose(ppg.c.signal);
% tt.PPG_D = transpose(ppg.d.signal);
% tt.PPG_E = transpose(ppg.e.signal);
% tt.PPG_F = transpose(ppg.f.signal);
% tt.PPG_G = transpose(ppg.g.signal);
% tt.PPG_H = transpose(ppg.h.signal);
% tt.ECG = transpose(ecg.signal);

%%
S = ppg.a.signal(1:4000);
plot(S);
findpeaks(S, 128, "MinPeakDistance", 0.8);

%%

fs = 128;                   % Sampling frequency                    
T = 1/Fs;                   % Sampling period       
L = length(S);              % Length of signal
t = (0:L-1)*T;              % Time vector
N = size(t,1);
Y = fft(S);
       
NFFT=2^12;       
X=fft(S, NFFT);

Px=X.*conj(X)/(NFFT*L); %Power of each freq components 

fVals=fs*(0:NFFT/2-1)/NFFT;      
plot(fVals,Px(1:NFFT/2),'b','LineSmoothing','on','LineWidth',1);         
title('One Sided Power Spectral Density');       
xlabel('Frequency (Hz)')         
ylabel('PSD');

%%
beats = double(ppg.heartBeat.signal);
% tt.BEAT = zeros(length(tt.Time), 1);
for i = beats(1:100)
    tseries = addevent(tseries, tsdata.event('beat', i));
%     tt(i, 'BEAT').BEAT = 1;
end


%%
% Beats in unix timestamps
beats = datetime(ppg.heartBeat.unixTimeStamps/1000, 'ConvertFrom', 'epochtime');
tt.BEAT = zeros(length(tt.Time), 1);
for i = beats
    tt(i, 'BEAT').BEAT = 1;
end

%% 
plot(tt.Time, tt.ECG);
hold on
scatter(tt.Time, tt.BEAT.*tt.ECG);
hold off

