%ICEEMDAN
function [modes]=ICEEMDAN(x,Nstd,NR,MaxIter,SNRFlag)
desvio_x=std(x);
x=x/desvio_x;
[a,b]=size(x);
temp=zeros(b,1);
 modes=zeros(b,1);
 aux=zeros(a,b);
for i=1:NR
    white_noise{i}=randn(size(x));
end;
 
for i=1:NR
    modes_white_noise{i}=emd(white_noise{i},'display',0);
end;
for i=1:NR 
    xi=x+Nstd*modes_white_noise{i}(:,1)'/std(modes_white_noise{i}(:,1));
    [temp, o, it]=emd(xi,'MaxNumIMF',1,'SiftMaxIterations',MaxIter,'display',0);
    aux=aux+(xi-temp')/NR;
end;
modes= (x-aux)'; 
medias = aux; 
k=1;
aux=zeros(a,b);
es_imf = min(size(emd(medias(1,:),'SiftMaxIterations',MaxIter,'display',0)));
while es_imf>1
    for i=1:NR
        tamanio=size(modes_white_noise{i});
        if tamanio(2)>=k+1
            noise=modes_white_noise{i}(:,k+1);
            if SNRFlag == 2
                noise=noise/std(noise); 
            end;
            noise=Nstd*noise;
            try
                [temp,o,it]=emd(medias(1,:)+std(medias(1,:))*noise','MaxNumIMF',1,'SiftMaxIterations',MaxIter,'display',0);
            catch    
                temp=emd(medias(1,:)+std(medias(1,:))*noise','MaxNumIMF',1,'SiftMaxIterations',MaxIter,'display',0);
            end;
        else
            try
                [temp, o, it]=emd(medias(1,:),'MaxNumIMF',1,'SiftMaxIterations',MaxIter,'display',0);
            catch
                temp=emd(medias(1,:),'MaxNumIMF',1,'SiftMaxIterations',MaxIter,'display',0);
            end;
        end;
        aux=aux+(medias(1,:)+std(medias(1,:))*noise'-temp')/NR;% r2 r3 r...
    end;
    modes=[modes (medias(1,:)-aux)'];
    medias = aux;
    aux=zeros(size(x));
    k=k+1;
    es_imf = min(size(emd(medias(1,:),'SiftMaxIterations',MaxIter,'display',0)));
end;
modes = [modes (medias(1,:))'];
modes=modes*desvio_x;


%PE
function pe = PE(data, m,t)
data = data(:); 
N = length(data); 
permlist = perms(1:m);
[h,~]=size(permlist);
c(1:length(permlist))=0; 
for j=1:N-t*(m-1) 
[~,iv(j,:)]=sort (data(j:t:j+t*(m-1)));
for jj=1:h
if (abs(permlist (jj,:)-iv(j,:)))==0
c(jj)= c(jj)+ 1; 
end
end
end
c=c(c~=0);
p= c/sum(c); 
pe=-sum(p.* log2(p));
pe=pe/log2(factorial(m));
end

%1D CNN BiLSTM
output = data1(:, end);
input = data1(:, 1:48);
[~, PSx] = mapminmax(input');
[~, PSy] = mapminmax(output');
numFolds = 5;
foldSize = floor(size(input, 1) / numFolds);
predictions1 = zeros(size(output));
numFeatures = size(input,2);
numHiddenUnits = 30;
numResponses = 1;
layers = [sequenceInputLayer(numFeatures)
          convolution1dLayer(1,32)
          bilstmLayer(numHiddenUnits)
          fullyConnectedLayer(numResponses)
          regressionLayer];
miniBatchSize = 64;
options = trainingOptions('adam', ...
                          'ExecutionEnvironment', 'cpu', ...
                          'MaxEpochs', 500, ...
                          'MiniBatchSize', miniBatchSize, ...
                          'GradientThreshold', 1, ...
                          'InitialLearnRate', 0.01, ...
                          'LearnRateSchedule', 'piecewise', ...
                          'LearnRateDropPeriod', 250, ...
                          'LearnRateDropFactor', 0.2, ...
                          'Verbose', false, ...
                          'Plots', 'training-progress');
for fold = 1:numFolds
    trainIdx = setdiff(1:size(input, 1), ((fold-1)*foldSize+1):(fold*foldSize));
    testIdx = ((fold-1)*foldSize+1):(fold*foldSize);
    xTrain = input(trainIdx, :);
    yTrain = output(trainIdx, :);
    xTest = input(testIdx, :);
    yTest = output(testIdx, :);
    XTrain = mapminmax('apply', xTrain', PSx);
    YTrain = mapminmax('apply', yTrain', PSy);
    XTest = mapminmax('apply', xTest', PSx);
    net1 = trainNetwork(XTrain, YTrain, layers, options);
    yTest_pre = predict(net1, XTest, 'MiniBatchSize', miniBatchSize, 'SequenceLength', 'longest');
    yTest_pre = mapminmax('reverse', yTest_pre, PSy);
    predictions(testIdx) = yTest_pre;
end

%ARIMAX
%ARIMAX uses the arima function of matlab.