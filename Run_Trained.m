%% Effect of training on face-selectivity in untrained networks (Fig.4) 
addpath('Data\PretrainedNet')
load('orderNetdesceding.mat')
load('IMG_cntr_210521.mat','IMG_cell','idx_mat')

%% Analysis for training effect (Fig 6b-f)
NN_type = 4;                                                               % number of types of networks. 1: "Untrained", 2: "Face-deprived", 3: "Original", 4: "Face"

if SimER == 1
    NN_trained = 3;                                                            % number of trained networks used in the anaylsis, N = 10 in the manuscript
    reN = 10;                                                                  % number of repetition for SVM training, N = 100 in the manuscript
    num_minTotal = 100;                                                        % number of randomly sampled units, all units was used in the manuscript
    
    Cell_Idx_trained = cell(NN_type,NN_trained);
    Cell_FSI_trained = cell(NN_type,NN_trained);
    Cell_SVM_trained = cell(NN_type,NN_trained);
    
    %% Training effect of face-units (Fig 6b-d)
    for nn = 1:NN_trained
        tic
        disp(['Trial',num2str(nn),' - Net',num2str(orderNetdesceding(nn))])
        for tt = 1:NN_type
            %% Load networks
            switch tt
                case 1
                    disp('Untrained')
                    dirNet = 'Data\PretrainedNet\Net_Untrained';
                    load([dirNet,'\Net_N',num2str(orderNetdesceding(nn)),'_std1.mat'],'net_rand');
                    act_rand = activations(net_rand,IMG_ORI,layersSet{length(layerArray)});
                    
                case 2
                    disp('Face-deprived')
                    dirNet = 'Data\PretrainedNet\Net_FD_ImageNet';
                    load([dirNet,'\Result_ImageNet_FD_Net',num2str(orderNetdesceding(nn)),'_210106.mat'],'net'); net_rand = net;
                    act_rand = activations(net_rand,IMG_ORI,layersSet{length(layerArray)});
                    
                case 3
                    disp('Original')
                    dirNet = 'Data\PretrainedNet\Net_ImageNet';
                    load([dirNet,'\Result_ImageNet_Net',num2str(orderNetdesceding(nn)),'_210106.mat'],'net'); net_rand = net;
                    act_rand = activations(net_rand,IMG_ORI,layersSet{length(layerArray)});
                    
                case 4
                    disp('ImageNet+Face')
                    dirNet = 'Data\PretrainedNet\Net_ImageNetwFace';
                    load([dirNet,'\Result_ImageNetwFace_Net',num2str(orderNetdesceding(nn)),'_210106.mat'],'net'); net_rand = net;
                    act_rand = activations(net_rand,IMG_ORI,layersSet{length(layerArray)});
            end
            
            %% Find face units
            num_cell = prod(array_sz(layerArray(length(layerArray)),:));
            [cell_idx] = fun_FindNeuron(act_rand,num_cell,numCLS,numIMG,pThr,idxClass);
            Cell_Idx_trained{tt,nn} = cell_idx;
            
            %% Caculate face-selectivity
            [~,rep_mat_3D] = fun_ResZscore(act_rand,num_cell,cell_idx,numCLS,numIMG);
            fsi_mat = fun_FSI(rep_mat_3D);
            Cell_FSI_trained{tt,nn} = fsi_mat;
            
            %% Perform face detection task
            array_SVM = zeros(reN,1);
            for rr = 1:reN
                [array_SVM(rr),~] = fun_SVM(net_rand,num_cell,cell_idx,IMG_cell,idx_mat,layersSet,length(layerArray),num_minTotal);
            end
            Cell_SVM_trained{tt,nn} = array_SVM;
        end
        toc
    end
    clearvars IMG_cell idx_mat
    
    array_num = zeros(NN_trained,NN_type);
    array_fsi = zeros(NN_trained,NN_type);
    array_svm = zeros(NN_trained,NN_type);
    
    for nn = 1:NN_trained
        for tt = 1:NN_type
            array_num(nn,tt) = length(Cell_Idx_trained{tt,nn});
            array_fsi(nn,tt) = nanmean(Cell_FSI_trained{tt,nn});
            array_svm(nn,tt) = mean(Cell_SVM_trained{tt,nn});
        end
    end
end

if SimER == 0
    load('Data_Trained.mat')
end

figure('units','normalized','outerposition',[0 0.5 1 0.5]); drawnow
sgtitle('Effect of training on face-selectivity in untrained networks (Fig 4b-d)')
subplot(2,6,[1,2,7,8])
boxplot([array_fsi(:,1),array_fsi(:,2),array_fsi(:,3),array_fsi(:,4)])
xticks([1:4]); ylim([0.3 0.5])
xticklabels({'Untrained','Face-deprived','Original','Face'}); ylabel('Face-selectivity index');

subplot(2,6,[3,4,9,10])
boxplot([array_num(:,1),array_num(:,2),array_num(:,3),array_num(:,4)])
xticks([1:4]); ylim([100 600])
xticklabels({'Untrained','Face-deprived','Original','Face'}); ylabel('Number of face units');

subplot(2,6,[5,6,11,12])
boxplot([array_svm(:,1),array_svm(:,2),array_svm(:,3),array_svm(:,4)])
xticks([1:4]); ylim([0.85 1.05])
xticklabels({'Untrained','Face-deprived','Original','Face'}); ylabel('Correct ratio');
