function [Outs_Final,Outs_MFCC,Outs_Sadjadi,Outs_New,t] = VAD_Drugman(wave,Fs,no)

% Voice activity detecion.
%
%
% Description
%  This is the voice activity detection (VAD) algorithm which is described
%  in [1]. The VAD exploits 3 sets of features: MFCCs as filter-based
%  features, and 2 sets of source-related features: 4 features presented in
%  Sadjadi's paper [2] and 3 proposed features.
%
% Inputs
%  wave            : [samples] [Nx1] input signal (speech signal)
%  Fs              : [Hz]      [1x1] sampling frequency
%  doPlot          : flag to show the results (=0 no plot, else plot the
%                    results). Default value=1.
%   
% Outputs
%  [Outs_Final,Outs_MFCC,Outs_Sadjadi,Outs_New]  : these are 4 vectors containing
%                   the VAD posteriors using respectively: i) the combined
%                   system which makes use of a decision fusion strategy
%                   and is based on the 3 feature sets, ii) the system
%                   using only MFCCs, iii) the system using only Sadjadi's
%                   features, iv) the system using only the proposed
%                   features.
%  t            : [seconds] Instants of the VAD posteriors.
%
% Example
%  Please see the Demo_VAD.m example file.
%
% References
%  [1] T.Drugman, Y. Stylianou, Y. Kida, M. Akamine: "Voice Activity Detection:
%  Merging Source and Filter-based Information", IEEE Signal Processing Letters,
%  2014.
%  [2] S.O. Sadjadi, J. Hansen: "Unsupervised Speech Activity Detection Using
%  Voicing Measures and Perceptual Spectral Flux", IEEE Sig. Pro. Letters,
%  vol. 20, pp. 197-200, 2013.
%
% Copyright (c) 2014 Toshiba Cambridge Research Laboratory
%
% License
%  This code will be part of the GLOAT toolbox (http://tcts.fpms.ac.be/~drugman/Toolbox/)
%  with the following licence:
%  This program is free software: you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation, either version 3 of the License, or
%  (at your option) any later version.
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
%
% This function will also be part of the Covarep project: http://covarep.github.io/covarep
% 
% Author 
%  Thomas Drugman thomas.drugman@umons.ac.be

if max(abs(wave)>1)
    wave=wave/max(abs(wave));
end

% Feature Extraction
[Feat,t] = VAD_Drugman_FeatureExtraction(wave,Fs,no);

% The feature trajectories are smoothed using a median-filter and the first
% and second derivatives are appended
Feat_MFCC=Feat(1:13,:);
[Feat_MFCC] = Feature_FilteringAndDerivatives(Feat_MFCC);
Feat_M = Feat_MFCC';
savepath = strcat('Test_feat_M',int2str(no));
save(savepath,'Feat_M')
Feat_Sadjadi=Feat(14:17,:);
[Feat_Sadjadi] = Feature_FilteringAndDerivatives(Feat_Sadjadi);
Feat_S = Feat_Sadjadi';
savepath = strcat('Test_feat_S',int2str(no));
save(savepath,'Feat_S')
Feat_New=Feat(18:20,:);
[Feat_New] = Feature_FilteringAndDerivatives(Feat_New);
Feat_N = Feat_New';
savepath = strcat('Test_feat_N',int2str(no));
save(savepath,'Feat_N')
Feat_All = [Feat_M; Feat_S; Feat_N];
savepath = strcat('Testing_feat_noise3_SNR1_',int2str(no));
save(savepath,'Feat_All')

% Normalization of MFCCs
load(['Minis_MFCC.mat'])
load(['Maxis_MFCC.mat'])
load(['ANNSystem_MFCC.mat'])
X=Feat_MFCC';
for k=1:size(X,1)
    vec=X(k,:);
    mini=Minis(k);
    maxi=Maxis(k);
    X(k,:)=-1+((X(k,:)-mini)/(maxi-mini))*2;
end


% Normalization of Sadjadi's features
load(['Minis_Sadjadi.mat'])
load(['Maxis_Sadjadi.mat'])
load(['ANNSystem_Sadjadi.mat'])
X=Feat_Sadjadi';
for k=1:size(X,1)
    vec=X(k,:);
    mini=Minis(k);
    maxi=Maxis(k);
    X(k,:)=-1+((X(k,:)-mini)/(maxi-mini))*2;
end


% Normalization of the new features
load(['Minis_New.mat'])
load(['Maxis_New.mat'])
load(['ANNSystem_New.mat'])
X=Feat_New';
for k=1:size(X,1)
    vec=X(k,:);
    mini=Minis(k);
    maxi=Maxis(k);
    X(k,:)=-1+((X(k,:)-mini)/(maxi-mini))*2;
end


function [MatFeat] = Feature_FilteringAndDerivatives(Feat)

Nfeats=size(Feat,1);
Nframes=size(Feat,2);

%Median filtering
for k=1:Nfeats
    Feat(k,:)=medfilt1(Feat(k,:),11);
end

MatFeat=zeros(Nframes,3*Nfeats);
MatFeat(:,1:Nfeats)=Feat';

% Computing the derivatives
Naround=10;
for k=1:Nfeats
    for l=Naround+1:Nframes
        Vec=Feat(k,l-Naround:l);
        Val=0;
        for p=1:length(Vec)-1
            Val=Val+(Vec(end)-Vec(p));
        end
        MatFeat(l,k+Nfeats)=Val;
    end
end

for k=1:Nfeats
    for l=Naround+1:Nframes
        Vec=MatFeat(l-Naround:l,k+Nfeats);
        Val=0;
        for p=1:length(Vec)-1
            Val=Val+(Vec(end)-Vec(p));
        end
        MatFeat(l,k+2*Nfeats)=Val;
    end
end