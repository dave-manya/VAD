clc

% filepath1 = 'C:\Users\hp\Desktop\New folder\VAD\VAD Database Final\WSJ training database\10 dB\';
% 
% xx = [];
% for i = 1:50
%     [x,Fs] = audioread(strcat(filepath1,int2str(i),'.wav'));
%     xx = [xx; x];
% end
% 
% filepath_store = 'C:\Users\hp\Desktop\New folder\VAD\VAD Database Final\WSJ training database\';
% audiowrite(strcat(filepath_store,'train.wav'),xx,Fs)


% filepath2 = 'C:\Users\hp\Desktop\New folder\VAD\VAD Database Final\ITU-T test database\clean speech data annotation_10ms frames\frame+';
% test_label = [];
% for j = 1:16
%     y = load(strcat(filepath2,int2str(j),'.mat'));
%     y = cell2mat(struct2cell(y));
%     y = y(1:length(y)-2);
%     test_label = [test_label y];
% end
% save('testing_labels','test_label')
% 
% 
 filepath3 = 'C:\Users\hp\Desktop\New folder\VAD\VAD Database Final\Codes\Drugman original package\Package\Testing_feat_noise2_SNR8_';
 testing_feat = [];
 for k = 1:16
     y = load(strcat(filepath3,int2str(k),'.mat'));
     y = cell2mat(struct2cell(y));
     testing_feat = [testing_feat y];
 end
 save('Testing_feat_noise2_SNR8','testing_feat')
 
% load('train_feat_MFCC500.mat')
% X_M=train_feat;
% min_M = min(X_M');
% save('min_MFCC500','min_M')
% max_M = max(X_M');
% save('max_MFCC500','max_M')
% for k=1:size(X_M,1)
%     vec=X_M(k,:);
%     mini=min_M(k);
%     maxi=max_M(k);
%     X_M(k,:)=-1+((X_M(k,:)-mini)/(maxi-mini))*2;
% end
% save('train_feat_normal_MFCC500','X_M')
% 
% load('train_feat_Sadjadi500.mat')
% X_S=train_feat;
% min_S = min(X_S');
% save('min_Sadjadi500','min_S')
% max_S = max(X_S');
% save('max_Sadjadi500','max_S')
% for k=1:size(X_S,1)
%     vec=X_S(k,:);
%     mini=min_S(k);
%     maxi=max_S(k);
%     X_S(k,:)=-1+((X_S(k,:)-mini)/(maxi-mini))*2;
% end
% save('train_feat_normal_Sadjadi500','X_S')
% 
% load('train_feat_New500.mat')
% X_N=train_feat;
% min_N = min(X_N');
% save('min_New500','min_N')
% max_N = max(X_N');
% save('max_New500','max_N')
% for k=1:size(X_N,1)
%     vec=X_N(k,:);
%     mini=min_N(k);
%     maxi=max_N(k);
%     X_N(k,:)=-1+((X_N(k,:)-mini)/(maxi-mini))*2;
% end
% save('train_feat_normal_New500','X_N')
% 
% % minis500 = [min_M min_S min_N];
% % save('minis500','minis500')
% % maxis500 = [max_M max_S max_N];
% % save('maxis500','maxis500')
% train_feat_normal500 = [X_M; X_S; X_N];
% save('train_feat_normal500','train_feat_normal500')
% 
% 
% load('Feat_M25.mat')
% X_M = Feat_M;
% for k=1:size(X_M,1)
%     vec=X_M(k,:);
%     mini=min_M(k);
%     maxi=max_M(k);
%     X_M(k,:)=-1+((X_M(k,:)-mini)/(maxi-mini))*2;
% end
% save('test_normal_M25','X_M')
% 
% load('Feat_S25.mat')
% X_S = Feat_S;
% for k=1:size(X_S,1)
%     vec=X_S(k,:);
%     mini=min_S(k);
%     maxi=max_S(k);
%     X_S(k,:)=-1+((X_S(k,:)-mini)/(maxi-mini))*2;
% end
% save('test_normal_S25','X_S')
% 
% load('Feat_N25.mat')
% X_N = Feat_N;
% for k=1:size(X_N,1)
%     vec=X_N(k,:);
%     mini=min_N(k);
%     maxi=max_N(k);
%     X_N(k,:)=-1+((X_N(k,:)-mini)/(maxi-mini))*2;
% end
% save('test_normal_N25','X_N')
% 
% test_normal = [X_M; X_S; X_N];
% save('test_normal25','test_normal')