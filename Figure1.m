%% Figure 1 : Spontaneous emergence of face-selectivity in untrained networks

%% DNN with randomly initialized weights (Fig 1b)
net_rand = Cell_Net{1,1,1};                                                                                 % untrained AlexNet
layers_ind = [2,10,14];                                                                                     % target layers: conv1~5
vis_N = 3;                                                                                                  % number of filters to visualize

figure('units','normalized','outerposition',[0 0.5 0.5 0.5]); colormap(gray)
sgtitle('Figure 1 : DNN with randomly initialized weights (Fig 1b)')
for ii = 1:length(layers_ind)
    filter_pre = net.Layers(layers_ind(ii)).Weights; ...
        sztmp = size(filter_pre,4); indtmpp=randi(sztmp, [1,vis_N]);                                        % load pre-trained filter
    filter_perm = net_rand.Layers(layers_ind(ii)).Weights; ...
        sztmp_r = size(filter_perm,4); indtmpp_r=randi(sztmp_r, [1,vis_N]);                                 % load permutedre filter
    for jj = 1:length(indtmpp)
        ax=subplot(3,7,7*(ii-1)+jj);
        tmp = squeeze(filter_pre(:,:,1,indtmpp(jj))); 
        imagesc(squeeze(filter_pre(:,:,1,indtmpp(jj)))); axis image off;                                    % plot pre-trained filter
        if jj == 2; title(['Pretrained Conv',num2str(2*ii-1),' filters']); end
        ax=subplot(3,7,7*(ii-1)+jj+4);
        tmp = squeeze(filter_perm(:,:,1,indtmpp_r(jj))); 
        imagesc(squeeze(filter_perm(:,:,1,indtmpp(jj)))); axis image off;                                    % plot permutedre filter
        if jj == 2; title(['Untrained Conv',num2str(2*ii-1),' filters']); end
    end
end

%% Simiarity-controlled stimulus set (Fig 1c, S1b, S1g)
figure('units','normalized','outerposition',[0 0 1 0.5]);
sgtitle('Figure 1 : Simiarity-controlled stimulus set (Fig 1c)')
% Figure 1c: Stimulus
posList = [1 2 3 9 10 8];
for ii = 1:numCLS
    subplot(2,7,posList(ii)); 
    imagesc(IMG_ORI(:,:,1,(ii-1)*numIMG+1)); colormap(gray); axis image off;
    title(STR_LABEL{ii})
end

% Figure S1b: Mean luminance of image
array_lum = zeros(numCLS,numIMG);
for cc = 1:numCLS
    for ii = 1:numIMG
        array_lum(cc,ii) = mean(IMG_ORI(:,:,1,(cc-1)*numIMG+ii)./255,'all');
    end
end

subplot(2,7,[4,5,11,12]); 
for cc = 1:numCLS
    hold on 
    bar([cc],mean(array_lum(cc,:)),'facecolor',[0.7 0.7 0.7])
    errorbar([cc],mean(array_lum(cc,:)),std(array_lum(cc,:)),'k')
end
xticks([1:numCLS]); xticklabels(STR_LABEL); ylim([0 1]); ylabel('Luminance'); title('Mean luminance of image (Fig S1b)')

% Figure S1g: Intra-class image similarity 
disp(['Measure intra-class image similarity ... (~ 50 sec)'])
tic
array_intraSim = zeros(numCLS,numIMG*(numIMG-1)/2);
for cc = 1:numCLS
    iStep = 0;
    for ii = 1:numIMG
        tmpIMG1 = IMG_ORI(:,:,1,ii+(cc-1)*numIMG); tmpIMG1 = tmpIMG1(:);
        for jj = ii:numIMG
            if jj == ii; continue; end
            iStep = iStep+1;
            tmpIMG2 = IMG_ORI(:,:,1,jj+(cc-1)*numIMG); tmpIMG2 = tmpIMG2(:);
            [r] = corrcoef(tmpIMG1,tmpIMG2);
            array_intraSim(cc,iStep) = r(1,2);
        end
    end
end
toc

subplot(2,7,[6,7,13,14]); hold on 
boxplot(array_intraSim'); ylim([-1 1]); xticklabels(STR_LABEL); ylabel('Image similarity'); title('Intra-class image similarity (Fig S1b)')

%% Face units in an untrained network (Fig 1d,e)
figure('units','normalized','outerposition',[0 0.5 1 0.5]);
sgtitle('Figure 1 : Face units in an untrained network (Fig 1d,e)')

% Figure 1d: Face-selective response
num_cell = prod(array_sz(layerArray(5),:));
net_rand = Cell_Net{1,1,1}; Idx_Face = Cell_Idx{1,1,1,5};

act_rand = activations(net_rand,IMG_ORI,layersSet{layerArray(5)});
[rep_mat,rep_mat_3D] = fun_ResZscore(act_rand,num_cell,Idx_Face,numCLS,numIMG);

subplot(2,6,[1,2,7,8]); load('Colorbar_Tsao.mat');
imagesc(rep_mat); caxis([-3 3])
for cc = 1:numCLS-1
line([numIMG*cc numIMG*cc], [1 length(Idx_Face)],'color','k','LineStyle','--')
end
xticks([100:numIMG:1100]); xticklabels(STR_LABEL); c = colorbar; colormap(cmap); %c.Label.String = 'Response (z-scored)';
ylabel('Unit indices'); title('Responses of face-selective units (Fig 1d)'); clearvars cmap

% Figure 1e: Tuning curve
subplot(2,6,[3,4,9,10]); hold on;
for ii = 1:length(Idx_Face)
    plot([0:numCLS+1],[0 mean(rep_mat_3D(ii,:,:),3) 0],'color',[0.7 0.7 0.7])
end
s = plot([0:numCLS+1],[0 mean(mean(rep_mat_3D,1),3) 0],'color','r');
xlim([0.5 numCLS+0.5]); xticks([1:numCLS]); xticklabels(STR_LABEL);
ylabel('Response (z-scored)'); title('Tuning curve of face-units'); legend(s,'Averaged','Location','northeast')

% Figure 1f: Face-selectivity index (FSI)
load('IMG_obj_Tsao2006_2010.mat'); IMG_Tsao = IMG; clearvars IMG % Tsao stimlus 
act_rand_Tsao = activations(net_rand,IMG_Tsao,layersSet{layerArray(5)});
[~,rep_Tsao_mat_3D] = fun_ResZscore(act_rand_Tsao,num_cell,Idx_Face,6,16);
rep_Tsao_shuf_mat_3D = reshape(rep_Tsao_mat_3D(randperm(numel(rep_Tsao_mat_3D))),size(rep_Tsao_mat_3D));
fsi_mat = fun_FSI(rep_Tsao_mat_3D);
fsi_shuf_mat = fun_FSI(rep_Tsao_shuf_mat_3D);

subplot(2,6,[5,6,11,12]); hold on;
boxplot([fsi_mat,fsi_shuf_mat])
xticks([1:2]); xticklabels({'Untrained','Response shuffled'});
ylabel('Face-selectivity index (FSI)'); title('Single neuron tuning (Fig 1e)');

%% Face units in untrained networks when varying inital weight distribution (Fig 1f,g)
disp(['Find face unit in untrained networks when varying inital weight distribution ... (~ 4 min)'])
tic
for nn = 1:NN
    disp(['%%% Trial : ',num2str(nn),' (',num2str(nn),'/',num2str(NN),')'])
    for vv = 1:length(verArray)
        disp(['%% Version : ',verSet{vv},' (',num2str(vv),'/',num2str(length(verArray)),')'])
        for ss = 1:length(stdArray)
            if vv*ss == 1; continue; end
            disp(['% Weight variation : ',num2str(stdArray(ss)),' (',num2str(ss),'/',num2str(length(stdArray)),')'])
            net_rand = fun_Initializeweight(net,verArray(vv),stdArray(ss));
            
            for ll = length(layerArray)
                num_cell = prod(array_sz(layerArray(ll),:));
                act_rand = activations(net_rand,IMG_ORI,layersSet{layerArray(ll)});
                
                [cell_idx] = fun_FindNeuron(act_rand,num_cell,numCLS,numIMG,pThr,idxClass);
                Cell_Idx{vv,ss,nn,ll} = cell_idx;
            end
            Cell_Net{vv,ss,nn} = net_rand; clearvars act_rand net_rand
        end
    end 
end

array_fsi_var = zeros(NN,length(verArray),length(stdArray));
array_num_var = zeros(NN,length(verArray),length(stdArray));
for nn = 1:NN
    for vv = 1:length(verArray)
        [stdArray_sort,stdArray_order] = sort(stdArray);
        for ss = 1:length(stdArray)
            net_rand = Cell_Net{vv,stdArray_order(ss),nn};
            Idx_Face = Cell_Idx{vv,stdArray_order(ss),nn,5};
            array_num_var(nn,vv,stdArray_order(ss)) = length(Idx_Face);
            
            act_rand = activations(net_rand,IMG_ORI,layersSet{layerArray(5)});
            [~,rep_mat_3D] = fun_ResZscore(act_rand,num_cell,Idx_Face,numCLS,numIMG);
            fsi_mat = fun_FSI(rep_mat_3D);
            array_fsi_var(nn,vv,stdArray_order(ss)) = nanmean(fsi_mat);
        end
    end
end
toc 

figure('units','normalized','outerposition',[0 0 1 0.5]);
sgtitle('Figure 1 : Face units in untrained networks when varying inital weight distribution (Fig 1f-g)')
% Figure : distribution of initial weight 
rand_layers_ind = [2,6,10,12,14];
tmpStdArray = [0.5 1 1.5];
tmpColorStd_Gau = [153/255 0 0; 1 0 0; 1 153/255 0]; 
tmpColorStd_Uni = [0 0 153/255; 0 0 1; 0 153/255 1];
edge = [-0.2:0.01:0.2];

subplot(2,6,[1,2]); hold on
for ss = 1:3
    tmpnet = Cell_Net{1,find(stdArray == tmpStdArray(ss)),1};
    W = [];
    for ll = 1:5
        targetlayer_ind = rand_layers_ind(1);
        weight_conv = tmpnet.Layers(targetlayer_ind ,1).Weights;
        fan_in = size(weight_conv,1)*size(weight_conv,2)*size(weight_conv,3);
        Wtmp = tmpStdArray(ss)*randn(size(weight_conv))*sqrt(1/fan_in);
        W = [W;Wtmp(:)];
    end
    
    tmp_rato = histcounts(W,edge,'Normalization', 'probability');
    plot(edge,[0 tmp_rato],'color',tmpColorStd_Gau(ss,:))
end
ylabel('Ratio'); xlim([-0.2 0.2]); ylim([0 0.2])
title('Distribution of intial weights (Gaussian)');legend('x0.5','x1','x1.5','Location','northeast')

subplot(2,6,[7,8]); hold on
for ss = 1:3
    tmpnet = Cell_Net{2,find(stdArray == tmpStdArray(ss)),1};
    W = [];
    for ll = 1:5
        targetlayer_ind = rand_layers_ind(1);
        weight_conv = tmpnet.Layers(targetlayer_ind ,1).Weights;
        fan_in = size(weight_conv,1)*size(weight_conv,2)*size(weight_conv,3);
        Wtmp = tmpStdArray(ss)*(rand(size(weight_conv))-0.5)*2*sqrt(3/fan_in);
    end
    W = [W; Wtmp(:)];
    tmp_rato = histcounts(W,edge,'Normalization', 'probability');
    plot(edge,[0 tmp_rato],'color',tmpColorStd_Uni(ss,:))
end
xlabel('Weight'); ylabel('Ratio'); xlim([-0.2 0.2]); ylim([0 0.2])
title('Distribution of intial weights (Uniform)'); legend('x0.5','x1','x1.5','Location','northeast');

% Figure 1f: variation of weights vs Number of face units
subplot(2,6,[3,4,9,10]); hold on
if NN == 1
    s1 = plot(stdArray_sort,squeeze(array_num_var(1,1,:)),'r');
    s2 = plot(stdArray_sort,squeeze(array_num_var(1,2,:)),'b');
else
    shadedErrorBar(stdArray_sort,squeeze(mean(array_num_var(:,1,:),1)),squeeze(std(array_num_var(:,1,:),1)),'lineprops','r');
    s1 = plot(stdArray_sort,squeeze(mean(array_num_var(:,1,:),1)),'r');
    shadedErrorBar(stdArray_sort,squeeze(mean(array_num_var(:,2,:),1)),squeeze(std(array_num_var(:,2,:),1)),'lineprops','b');
    s2 = plot(stdArray_sort,squeeze(mean(array_num_var(:,2,:),1)),'b');
end
ylim([0 600]); ylabel('Number of units'); xlabel('Variation of weights'); 
title('Number of face units (Fig 1f)'); legend([s1,s2],'Gaussian','Uniform','Location','northeast')

% Figure 1g: variation of weights vs FSI
subplot(2,6,[5,6,11,12]); hold on
if NN == 1
    s1 = plot(stdArray_sort,squeeze(array_fsi_var(1,1,:)),'r');
    s2 = plot(stdArray_sort,squeeze(array_fsi_var(1,2,:)),'b');
else
    shadedErrorBar(stdArray_sort,squeeze(mean(array_fsi_var(:,1,:),1)),squeeze(std(array_fsi_var(:,1,:),1)),'lineprops','r');
    s1 = plot(stdArray_sort,squeeze(mean(array_fsi_var(:,1,:),1)),'r');
    shadedErrorBar(stdArray_sort,squeeze(mean(array_fsi_var(:,2,:),1)),squeeze(std(array_fsi_var(:,2,:),1)),'lineprops','b');
    s2 = plot(stdArray_sort,squeeze(mean(array_fsi_var(:,2,:),1)),'b');
end
ylim([0 0.6]); ylabel('Face-selectivity index (FSI)'); xlabel('Variation of weights'); 
title('Single unit selectivity (Fig 1g)'); legend([s1,s2],'Gaussian','Uniform','Location','northeast')