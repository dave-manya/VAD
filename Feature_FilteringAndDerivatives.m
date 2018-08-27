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