%% Demo code ver. 11/01/2021
%==================================================================================================================================================
% Face Detection in Untrained Deep Neural Networks
% Seungdae Baek, Min Song, Jaeson Jang, Gwangsu Kim & Se-Bum Paik*
%
% *Contact: sbpaik@kaist.ac.kr
%
% Prerequirement 
% 1) MATLAB 2019b or later version is recommended.
% 2) Install deeplearning toolbox. 
% 3) Please download 'AlexNet_2018b.mat' (or pretrained AlexNet provided by MATLAB), 'Data.zip', 'Stimulus.zip'  from below link
%
%      - [Data URL] : 
%
%    and unzip these files in the same directory

% Output of the code
% Below results for untrained AlexNet will be shown.
% Result 1) Run_Unit: Spontaneous emergence of face-selectivity in untrained networks (Fig.1, Fig.S1-3)
% Result 2) Run_PFI: Preferred feature images of face-selective units in untrained networks (Fig.2, Fig.S4) 
% Result 3) Run_SVM: Detection of face images using the response of face units in untrained networks (Fig.3, Fig.S11-12) 
% Result 4) Run_Trained: Effect of training on face-selectivity in untrained networks (Fig.4) 
% Result 5) Run_Invariance: Invariant characteristics of face-selective units in untrained networks (Fig.S5) 
% Result 6) Run_View: Viewpoint invariance of face-selective units in untrained networks (Fig.S8) 
%==================================================================================================================================================
close all;clc;clear;
seed = 1; rng(seed)                                                       % fixed random seed for regenerating same result

addpath('Data')
addpath('Stimulus')
addpath('Subfunctions')
toolbox_chk;                                                               % checking matlab version and toolbox

tic

%% Setting parameters
% Demo code
res1 = 1; res2 = 0; res3 = 0; res4 = 0; res5 = 0; res6 = 1;                % flag for analysis corresponding each figure 
NN = 1;                                                                    % number of networks for analysis
 
% Image
STR_LABEL = {'Face','Hand','Horn','Flower','Chair','Scrambled'};           % label of classes in the object stimulus
numIMG = 200;                                                              % number of images of a class in the object stimulus
numCLS = 6;                                                                % number of classes in the object stimulus
inpSize = 227;                                                             % width or hieght of each image in the object stimulus

% Network
layersSet = {'relu1', 'relu2', 'relu3', 'relu4', 'relu5'};	               % names of feature extraction layers
array_sz = [55 55 96; 27 27 256; 13 13 384; 13 13 384; 13 13 256];         % dimensions of activation maps of each layer
layerArray = [1:5];                                                        % target layers
stdArray = [1 0.01 0.5 1.5 2];                                             % std of gaussian kernel for randomly initialized network
verSet = {'LeCun - Normal dist','LeCun - Uniform dist'};
verArray = [1 2];                                                          % version of initialization 
                                                                           %  1: LeCun / 2: LeCun uniform 
% Analysis                                                                           
pThr = 0.001;                                                              % p-value threshold of selective response
idxClass = 1;                                                              % index of face class in the dataset

%% Step 1. Loading pretrained Alexnet and image dataset
disp(['Load imageset and networks ... (~ 10 sec)'])
tic
net = alexnet;                                                             % pretained AlexNet
load('IMG_cntr_210521.mat','IMG')                                          % object stimulus

toc
IMG = IMG(:,:,1:numIMG*numCLS); IMG_ORI = single(repmat(permute(IMG,[1 2 4 3]),[1 1 3])); clearvars IMG
% IMG = IMG(:,:,:,1:numIMG*numCLS); IMG_ORI = single(IMG); clearvars IMG

disp(['Find face unit in untrained network ... (~ 30 sec)'])
Cell_Net = cell(length(verArray),length(stdArray),NN);
Cell_Idx = cell(length(verArray),length(stdArray),NN,length(layerArray));

for nn = 1:NN
    tic
    disp(['%%% Trial : ',num2str(nn),' (',num2str(nn),'/',num2str(NN),')'])
    for vv = 1
        disp(['%% Version : ',verSet{vv},' (',num2str(vv),'/',num2str(length(vv)),')'])
        for ss = 1
            disp(['% Weight variation : ',num2str(stdArray(ss)),' (',num2str(ss),'/',num2str(length(ss)),')'])
            %% Step 2. Loading and generating untrained AlexNet
            net_rand = fun_Initializeweight(net,verArray(vv),stdArray(ss));
            
            for ll = length(layerArray)
                %% Step 3. Measuring responses of neurons in the target layer
                num_cell = prod(array_sz(layerArray(ll),:));
                act_rand = activations(net_rand,IMG_ORI,layersSet{layerArray(ll)});
                
                %% Step 4. Finding selective neuron to target class
                [cell_idx] = fun_FindNeuron(act_rand,num_cell,numCLS,numIMG,pThr,idxClass);
                Cell_Idx{vv,ss,nn,ll} = cell_idx; clearvars cell_idx
            end
            Cell_Net{vv,ss,nn} = net_rand; clearvars act_rand net_rand
        end
    end 
    toc
end

if res1 == 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run_Unit: Spontaneous emergence of face-selectivity in untrained networks
disp('Result 1 ... (~ 10 min)')
tic
Run_Unit;
toc
end

if res2 == 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run_PFI: Preferred feature images of face-selective units in untrained networks 
disp('Result 2 ... (~ 2 min)')
tic
% Decide the simulation type
% 0 : Fast version of PFI simulation. The saved PFI would be displayed.
% 1 : Actual calculation process would be run. It takes around 30 minutes.
Sim = 0;
Figure2;
toc
end

if res3 == 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run_SVM: Detection of face images using the response of face units in untrained networks 
disp('Result 3 ... (~ 5 min)')
tic
Run_SVM;
toc
end

if res4 == 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run_Trained: Effect of training on face-selectivity in untrained networks
disp('Result 4 ... (~ 5 min)')
tic
% 0 : Loading data in our manuscript (Number of networks = 10, in manuscript)
% 1 : Fast version of analysis for training effect (Number of networks = 3)
SimER = 0;
toc
end

if res5 == 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run_Invariance: Invariant characteristics of face-selective units in untrained networks
disp('Result 5 ... (~ 5 min)')
tic
% Decide the simulation type (TT)
% 1 : Fast version of invariance analysis. Results for translation invariance would be displayed.
% 3 : Result for translation, size, rotation invariance would be displayed. It takes around 10 minutes.
TT = 1;
Run_Invariance;
toc
end

if res6 == 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run_View: Viewpoint invariance of face-selective units in untrained networks
disp('Result 6 ... (~ 5 min)')
tic
Run_View;
toc
end
