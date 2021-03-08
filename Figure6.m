%% Figure 6 : Effect of training on face-selectivity in untrained networks
addpath('PretrainedNet')
load('orderNetdesceding.mat')
load('IMG_obj_201201.mat','IMG_cell','idx_mat')

%% Analysis for training effect (Fig 6b-f)
NN_trained = 3;                                                            % number of trained networks used in the anaylsis, N = 10 in the manuscript
NN_type = 4;                                                               % number of types of networks. 1: "Untrained", 2: "Face-deprived", 3: "Original", 4: "Face"
reN = 10;                                                                  % number of repetition for SVM training, N = 100 in the manuscript
num_minTotal = 100;                                                        % number of randomly sampled units, all units was used in the manuscript

Cell_Idx_trained = cell(NN_type,NN_trained);
Cell_FSI_trained = cell(NN_type,NN_trained);
Cell_SVM_trained = cell(NN_type,NN_trained);
Cell_ER_trained = cell(NN_type,NN_trained,3);
Cell_var_axis = cell(3,1);

%% Training effect of face-units (Fig 6b-f)
for nn = 1:NN_trained
    tic
    disp(['Trial',num2str(nn),' - Net',num2str(orderNetdesceding(nn))])
    for tt = 1:NN_type
        %% Load networks
        switch tt
            case 1
                disp('Untrained')
                dirNet = 'PretrainedNet\Net_Untrained';
                load([dirNet,'\Net_N',num2str(orderNetdesceding(nn)),'_std1.mat'],'net_rand');
                act_rand = activations(net_rand,IMG_ORI,layersSet{length(layerArray)});
                
            case 2
                disp('Face-deprived')
                dirNet = 'PretrainedNet\Net_FD_ImageNet';
                load([dirNet,'\Result_ImageNet_FD_Net',num2str(orderNetdesceding(nn)),'_210106.mat'],'net'); net_rand = net;
                act_rand = activations(net_rand,IMG_ORI,layersSet{length(layerArray)});
                
            case 3
                disp('Original')
                dirNet = 'PretrainedNet\Net_ImageNet';
                load([dirNet,'\Result_ImageNet_Net',num2str(orderNetdesceding(nn)),'_210106.mat'],'net'); net_rand = net;
                act_rand = activations(net_rand,IMG_ORI,layersSet{length(layerArray)});
                
            case 4
                disp('ImageNet+Face')
                dirNet = 'PretrainedNet\Net_ImageNetwFace';
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
        
        %% Measure effective range of invariance 
        if SimER == 1
            for vtype = 1:3
                switch vtype
                    case 1
                        disp('Position')
                        load('IMG_var_pos_201201.mat'); IMG_var = single(IMG_pos); clearvars IMG_pos
                        var_idx = pos_idx; clearvars pos_idx
                        RF_size = 163/2;
                        var_axis = (-120:20:120)/RF_size;
                    case 2
                        disp('Size')
                        load('IMG_var_size_201201.mat'); IMG_var = single(IMG_size); clearvars IMG_size
                        var_idx = size_idx; clearvars size_idx
                        RF_size = 163;
                        var_axis = (41:25:341)/RF_size*100;
                    case 3
                        disp('Rotation')
                        load('IMG_var_rot_201201.mat'); IMG_var = single(IMG_rot); clearvars IMG_rot
                        var_idx = rot_idx; clearvars rot_idx
                        RF_size = 1;
                        var_axis = -180:30:180;
                end
                Cell_var_axis{vtype} = var_axis;
                
                % Measure network response
                act_rand = activations(net_rand,IMG_var,layersSet{layerArray(length(layerArray))});
                act_re = reshape(act_rand,num_cell,size(IMG_var,4));
                act_face = act_re(cell_idx,:);
                num_face_cell = size(cell_idx,1);
                clearvars act_rand act_re
                
                % Measure effective range
                [resp_z_mat,face_resp_z_mat,max_resp_z_mat,ER_single_mat] = fun_EffectRange_Resp(act_face,num_face_cell,cls_idx,var_idx);
                Cell_ER_trained{tt,nn,vtype} = ER_single_mat;
            end
        end
    end
    toc
end
clearvars IMG_cell idx_mat

array_num = zeros(NN_trained,NN_type);
array_fsi = zeros(NN_trained,NN_type); 
array_svm = zeros(NN_trained,NN_type); 
array_er = zeros(NN_trained,NN_type,3); 

for nn = 1:NN_trained
    for tt = 1:NN_type
        array_num(nn,tt) = length(Cell_Idx_trained{tt,nn});
        array_fsi(nn,tt) = nanmean(Cell_FSI_trained{tt,nn});
        array_svm(nn,tt) = mean(Cell_SVM_trained{tt,nn});
        for vv = 1:3
            array_er(nn,tt,vv) = mean(Cell_ER_trained{tt,nn});
        end
    end
end

arrayYlim = [3, 300, 400];
StrYlabel = {'Effective range (r_R_F)','Effective range (%)','Effective range (deg)'};

if SimER == 0
    figure('units','normalized','outerposition',[0 0.5 1 0.5]); drawnow
    sgtitle('Figure 6 : Effect of training on face-selectivity in untrained networks (Fig 6b-f)')
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
else
    figure('units','normalized','outerposition',[0 0 1 1]); drawnow
    sgtitle('Figure 6 : Effect of training on face-selectivity in untrained networks (Fig 6b-f)')
    subplot(4,6,[1,2,7,8])
    boxplot([array_fsi(:,1),array_fsi(:,2),array_fsi(:,3),array_fsi(:,4)])
    xticks([1:4]); ylim([0.3 0.5])
    xticklabels({'Untrained','Face-deprived','Original','Face'}); ylabel('Face-selectivity index');
    
    subplot(4,6,[3,4,9,10])
    boxplot([array_num(:,1),array_num(:,2),array_num(:,3),array_num(:,4)])
    xticks([1:4]); ylim([100 600])
    xticklabels({'Untrained','Face-deprived','Original','Face'}); ylabel('Number of face units');
    
    subplot(4,6,[5,6,11,12])
    boxplot([array_svm(:,1),array_svm(:,2),array_svm(:,3),array_svm(:,4)])
    xticks([1:4]); ylim([0.85 1.05])
    xticklabels({'Untrained','Face-deprived','Original','Face'}); ylabel('Correct ratio');
    
    subplot(4,6,[13,14,19,20]); hold on; vtype = 1;
    tmpScale = Cell_var_axis{vtype}(end)-Cell_var_axis{vtype}(end-1);
    bar([1],mean(tmpScale.*array_er(:,1,vtype),1),0.5,'facecolor',[0 0 0])
    errorbar([1],mean(tmpScale.*array_er(:,1,vtype),1),std(tmpScale.*array_er(:,1,vtype),1),'k')
    bar([2],mean(tmpScale.*array_er(:,4,vtype),1),0.5,'facecolor',[0 0 1])
    errorbar([2],mean(tmpScale.*array_er(:,4,vtype),1),std(tmpScale.*array_er(:,4,vtype),1),'k')
    xticks([1:2]); xlim([0.5 2.5]); ylim([0 arrayYlim(vtype)])
    xticklabels({'Untrained','Trained-Face'});  ylabel(StrYlabel{vtype});
    title('Effect range of invariance - Translation (Fig 6f)')
    
    subplot(4,6,[15,16,21,22]); hold on; vtype = 2;
    tmpScale = Cell_var_axis{vtype}(end)-Cell_var_axis{vtype}(end-1);
    bar([1],mean(tmpScale.*array_er(:,1,vtype),1),0.5,'facecolor',[0 0 0])
    errorbar([1],mean(tmpScale.*array_er(:,1,vtype),1),std(tmpScale.*array_er(:,1,vtype),1),'k')
    bar([2],mean(tmpScale.*array_er(:,4,vtype),1),0.5,'facecolor',[0 0 1])
    errorbar([2],mean(tmpScale.*array_er(:,4,vtype),1),std(tmpScale.*array_er(:,4,vtype),1),'k')
    xticks([1:2]); xlim([0.5 2.5]); ylim([0 arrayYlim(vtype)])
    xticklabels({'Untrained','Trained-Face'});  ylabel(StrYlabel{vtype});
    title('Effect range of invariance - Scaling (Fig 6f)')
    
    subplot(4,6,[17,18,23,24]); hold on; vtype = 3;
    tmpScale = Cell_var_axis{vtype}(end)-Cell_var_axis{vtype}(end-1);
    bar([1],mean(tmpScale.*array_er(:,1,vtype),1),0.5,'facecolor',[0 0 0])
    errorbar([1],mean(tmpScale.*array_er(:,1,vtype),1),std(tmpScale.*array_er(:,1,vtype),1),'k')
    bar([2],mean(tmpScale.*array_er(:,4,vtype),1),0.5,'facecolor',[0 0 1])
    errorbar([2],mean(tmpScale.*array_er(:,4,vtype),1),std(tmpScale.*array_er(:,4,vtype),1),'k')
    xticks([1:2]); xlim([0.5 2.5]); ylim([0 arrayYlim(vtype)])
    xticklabels({'Untrained','Trained-Face'});  ylabel(StrYlabel{vtype});
    title('Effect range of invariance - Rotation (Fig 6f)')
end