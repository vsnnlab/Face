%% Demo code ver. 03/08/2021
%==================================================================================================================================================
% Face Detection in Untrained Deep Neural Networks
% Seungdae Baek, Min Song, Jaeson Jang, Gwangsu Kim & Se-Bum Paik*
%
% *Contact: sbpaik@kaist.ac.kr
%
% Prerequirement 
% 1) MATLAB 2019b or later version is recommended.
% 2) Install deeplearning toolbox. 
%
% Output of the code
% Below results for untrained AlexNet will be shown.
% 1) Spontaneous emergence of face-selectivity in untrained networks (Figure 1)
% 2) Preferred feature images of face-selective units in untrained networks (Figure 2) 
% 3) Detection of face images using the response of face units in untrained networks (Figure 3) 
% 4) Invariant characteristics of face-selective units in untrained networks  (Figure 4) 
% 5) Invariant characteristics of face-selective units in untrained networks  (Figure 5) 
% 6) Effect of training on face-selectivity in untrained networks  (Figure 6) 
%==================================================================================================================================================
close all;clc;clear;
seed = 97; rng(seed)                                                       % fixed random seed for regenerating same result

addpath('Dataset')
addpath('Dataset\stimulus')
addpath('Subfunctions')
toolbox_chk;                                                               % checking matlab version and toolbox

tic

%% Setting parameters
% Demo code
fig1 = 1; fig2 = 1; fig3 = 1; fig4 = 1; fig5 = 1; fig6 = 1;                % flag for analysis corresponding each figure 
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
load('AlexNet_2018b.mat')                                                  % pretained AlexNet
load('IMG_obj_201201.mat','IMG')                                           % object stimulus
toc
IMG = IMG(:,:,:,1:numIMG*numCLS); IMG_ORI = single(IMG); clearvars IMG

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

if fig1 == 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Figure 1 : Spontaneous emergence of face-selectivity in untrained networks
disp('Figure 1 ... (~ 5 min)')
tic
Figure1;
toc
end

if fig2 == 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Figure 2 : Preferred feature images of face-selective units in untrained networks
disp('Figure 2 ... (~ 2 min)')
tic
% Decide the simulation type
% 1 : Fast version of PFI simulation. The saved PFI would be displayed.
% 2 : Actual calculation process would be run. It takes around 30 minutes.
Sim = 1;
Figure2;
toc
end

if fig3 == 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Figure 3 : Detection of face images using the response of face units in untrained networks 
disp('Figure 3 ... (~ 5 min)')
tic
Figure3;
toc
end

if fig4 == 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Figure 4 : Invariant characteristics of face-selective units in untrained networks 
disp('Figure 4 ... (~ 3 min)')
tic
% Decide the simulation type (TT)
% 1 : Fast version of invariance analysis. Results for translation invariance would be displayed.
% 3 : Result for translation, size, rotation invariance would be displayed. It takes around 10 minutes.
TT = 1;
Figure4;
toc
end

if fig5 == 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Figure 5 : Increasing viewpoint invariance of face-selective units along the network hierarchies
disp('Figure 5 ... (~ 3 min)')
tic
Figure5;
toc
end

if fig6 == 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Figure 6 : Effect of training on face-selectivity in untrained networks
disp('Figure 6 ... (~ 5 min)')
tic
% Decide the simulation type (SimER)
% 0 : Fast version of invariance analysis. Results for training effect without invariance analysis would be displayed.
% 1 : All Results for training effect including invariance anaylsis would be displayed. It takes around 30 minutes.
SimER = 1;
Figure6;
toc
end