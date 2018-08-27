%% Code to produce delta and double delta features and to normalise the features
function [MatMFCC, MatSadj, MatNew]=find_delta_of_features(MatFeat,t)

addpath Auxilary

% Normalize features and derivatives
% MatFeat = MatFeat';
% featMin = min(MatFeat,[],2); featMax = max(MatFeat,[],2);
% tmp = (featMax - featMin); divisor = []; tmp_min = [];
% for i = 1:size(MatFeat, 2);
%     divisor = [divisor tmp];
%     tmp_min = [tmp_min featMin];
% end
% MatFeat = (MatFeat - tmp_min) ./ divisor;
% MatFeat = MatFeat';

[Feat] = Feature_FilteringAndDerivatives_mod(MatFeat);

%Normalise features afer finding derivatives
Feat = Feat';
featMin = min(Feat,[],2); featMax = max(Feat,[],2);
featMean = mean(Feat,2);
tmp = (featMax - featMin); divisor = []; tmp_mean = [];
for i = 1:size(Feat, 2);
    divisor = [divisor tmp];
    tmp_mean = [tmp_mean featMean];
end
Feat = (Feat - tmp_mean) ./ divisor;
Feat = Feat';

MatMFCC = [transpose(Feat(:,1:13)); transpose(Feat(:,21:33)); transpose(Feat(:,41:53))];
MatSadj = [transpose(Feat(:,14:17)); transpose(Feat(:,34:37)); transpose(Feat(:,54:57))];
MatNew = [transpose(Feat(:,18:20)); transpose(Feat(:,38:40)); transpose(Feat(:,58:60))];


end