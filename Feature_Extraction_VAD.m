function [] = Demo_VAD()

clc
% Read any speech audio file
total = 1500
filepath = 'Path to .wav file';
for no = 1:total
    [wave,Fs] = audioread(strcat(filepath,int2str(no),'.wav'));

    % VAD algorithm
    % The third input is just a flag to plot the results or not
    % The outputs are the following:
    % [Outs_Final,Outs_MFCC,Outs_Sadjadi,Outs_New]  : these are 4 vectors containing
    %                   the VAD posteriors using respectively: i) the combined
    %                   system which makes use of a decision fusion strategy
    %                   and is based on the 3 feature sets, ii) the system
    %                   using only MFCCs, iii) the system using only Sadjadi's
    %                   features, iv) the system using only the proposed
    %                   features.
    %                 t : [seconds] Instants of the VAD posteriors.

    [Outs_Final,Outs_MFCC,Outs_Sadjadi,Outs_New,t] = VAD_Drugman(wave,Fs,no);
end