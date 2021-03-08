%% Figure 5 : Increasing viewpoint invariance of face-selective units along the network hierarchies

%% Analysis for viewpoint invariance (Fig 5a,f-h)
net_rand = Cell_Net{1,1,1};                                                % untrained AlexNet  

load('IMG_var_view_210106.mat');                                           % viewpoint stimulus
IMG_view = single(IMG_viewpoint); clearvars IMG_viewpoint   
numIMG_view = 10;                                                          % number of images of a class in the object stimulus
numCLS_view = 5;                                                           % number of classes in the viewpoint stimulus          
arrayClass = [ones(numIMG_view,1);2.*ones(numIMG_view,1);3.*ones(numIMG_view,1);4.*ones(numIMG_view,1);5.*ones(numIMG_view,1)];

%% Find face units on each L3, L4
disp(['Find face units on L3, L4 ... (~ 2 min)'])
tic
for ll = 3:4
    num_cell = prod(array_sz(layerArray(ll),:));
    act_rand = activations(net_rand,IMG_ORI,layersSet{layerArray(ll)});
    [cell_idx] = fun_FindNeuron(act_rand,num_cell,numCLS,numIMG,pThr,idxClass);
    Cell_Idx{1,1,1,ll} = cell_idx; clearvars cell_idx act_rand
end
toc

%% Find viewpoint specfic / invariant face units
Cell_view_p = cell(5,1);
Cell_view_pref = cell(5,1);
Cell_view_peak = cell(5,1);
Cell_view_stat = cell(5,1);

for ll = 3:5
    indLayer  = ll;
    num_cell = prod(array_sz(indLayer,:));
    Idx_Face = Cell_Idx{1,1,1,ll};
    
    act_rand = activations(net_rand,IMG_view,layersSet{indLayer});
    act_reshape = reshape(act_rand,num_cell,size(IMG_view,4)); clearvars act_rand 
    act_reshape3D = zeros(num_cell,numCLS_view,numIMG_view);
    for cc = 1:numCLS_view
        act_reshape3D(:,cc,:) = act_reshape(:,(cc-1)*numIMG_view+1:cc*numIMG_view);
    end
    act_reshape = act_reshape(Idx_Face,:);
    act_reshape3D = act_reshape3D(Idx_Face,:,:); clearvars act_rand 
    
    p = zeros(length(Idx_Face),1); 
    pref = zeros(length(Idx_Face),numCLS_view); 
    peak = zeros(length(Idx_Face),numCLS_view); % peak location
    vstat = zeros(length(Idx_Face),1);
    
    for ii = 1:length(Idx_Face)
        p(ii) = anova1(act_reshape(ii,:),arrayClass,'off');
        
        meanFR = squeeze(mean(act_reshape3D(ii,:,:),3))';
        [~,peakView] = findpeaks([min(meanFR);meanFR;min(meanFR)]); peak(ii,peakView-1) = 1;
        
        if sum(meanFR) == 0
            pref(ii,:) = 0;
        else
            [~,pref(ii,:)] = sort(meanFR,'descend');
            vstat(ii,1) = std(meanFR);
        end
    end
    
    Cell_view_p{ll} = p;
    Cell_view_pref{ll} = pref;
    Cell_view_peak{ll} = peak;
    Cell_view_stat{ll} = vstat;
end

sel_cellview_mat = cell(length(layerArray),5); % index : inv / spe / spe_pre / mirro (-90, 90) / mirro (-45, 45)
for ll = 3:5
    idx = Cell_Idx{1,1,1,ll};
    p = Cell_view_p{ll};
    pref = Cell_view_pref{ll};
    peak = Cell_view_peak{ll};
    
    sel_cellview_mat{ll,1} = idx((pref(:,1) ~= 0)&(p(:,1)>=0.05));
    sel_cellview_mat{ll,2} = idx((pref(:,1) ~= 0)&(p(:,1)<0.05)&(sum(peak,2) == 1));
    sel_cellview_mat{ll,3} = pref((pref(:,1) ~= 0)&(p(:,1)<0.05)&(sum(peak,2) == 1));
    
    sel_cellview_mat{ll,4} = idx((pref(:,1) ~= 0)&(p(:,1)<0.05)&(sum(peak,2) > 1)&((pref(:,1) == 1)|(pref(:,1) == 5))&(pref(:,1) ~= 3));
    sel_cellview_mat{ll,5} = idx((pref(:,1) ~= 0)&(p(:,1)<0.05)&(sum(peak,2) > 1)&((pref(:,1) == 2)|(pref(:,1) == 4))&(pref(:,1) ~= 3));
end

figure('units','normalized','outerposition',[0 0 1 1]); drawnow
sgtitle(['Figure 5 : Increasing viewpoint invariance of face-selective units along the network hierarchies'])

% Viewpoint variation (Fig 5a)
pos_subplot = [1:5]; pos_img = [1:numIMG_view:41];
for ii = 1:5
    subplot(4,8,pos_subplot(ii)); imagesc(IMG_view(:,:,1,pos_img(ii))); axis image off; colormap(gray); caxis([0 255])
end
pos_subplot = [9:13]; pos_img = [2:numIMG_view:42];
for ii = 1:5
    subplot(4,8,pos_subplot(ii)); imagesc(IMG_view(:,:,1,pos_img(ii))); axis image off; colormap(gray); caxis([0 255]); title([num2str(-90+(ii-1)*45),' deg'])
end
    
% Face unit response (Conv5) (Fig 5a)
act_rand = activations(net_rand,IMG_view,layersSet{length(layerArray)});
act_reshape = reshape(act_rand,num_cell,size(IMG_view,4));
act_Norm = act_reshape./mean(act_reshape,2);
act_Norm3D = zeros(num_cell,numCLS_view,numIMG_view);
for cc = 1:numCLS_view
    act_Norm3D(:,cc,:) = act_Norm(:,(cc-1)*numIMG_view+1:cc*numIMG_view);
end

cell_list_inv = sel_cellview_mat{5,1};
cell_list_spe3 = sel_cellview_mat{5,2}(sel_cellview_mat{5,3} == 3);

subplot(4,8,[7,8,15,16]); hold on
shadedErrorBar([-90:45:90],mean(nanmean(act_Norm3D(cell_list_spe3,:,:),1),3),std(nanmean(act_Norm3D(cell_list_spe3,:,:),1),[],3)./sqrt(numIMG_view),'lineprops',{'b','markerfacecolor','b'});
shadedErrorBar([-90:45:90],mean(nanmean(act_Norm3D(cell_list_inv,:,:),1),3),std(nanmean(act_Norm3D(cell_list_inv,:,:),1),[],3)./sqrt(numIMG_view),'lineprops',{'r','markerfacecolor','r'});
s1 = plot([-90:45:90],mean(nanmean(act_Norm3D(cell_list_spe3,:,:),1),3),'color','b');
s2 = plot([-90:45:90],mean(nanmean(act_Norm3D(cell_list_inv,:,:),1),3),'color','r');
line([-90 90],[1 1],'Color','k','LineStyle','--')
legend([s1,s2],{'Viewpoint-specific (0 deg)','Viewpoint-invariant'},'Location','southeast')
xticks([-90:45:90]); xticklabels({'-90 deg','-45 deg','0 deg','45 deg','90 deg'}); xlim([-90 90]); 
ylim([0 mean(nanmean(act_Norm3D(cell_list_spe3,3,:),1),3)+std(nanmean(act_Norm3D(cell_list_spe3,3,:),1),[],3)./sqrt(numIMG_view)+0.2]); ylabel('Normalized Response (A.U.)'); title('Face unit response (Fig 5a)')

% Average tuning curves of face units in each layer (Fig 5f)
Cell_align_tuning = cell(3,1);
for ll = 1:3
    idx = Cell_Idx{1,1,1,ll+2}; num_cell = prod(array_sz(ll+2,:));
    act_rand = activations(net_rand,IMG_view,layersSet{ll+2});
    act_reshape = reshape(act_rand,num_cell,size(IMG_view,4)); clearvars act_rand 
    act_Norm = act_reshape;
    act_Norm3D = zeros(num_cell,numCLS_view,numIMG_view);
    for cc = 1:numCLS_view
        act_Norm3D(:,cc,:) = act_Norm(:,(cc-1)*numIMG_view+1:cc*numIMG_view);
    end
    act_Norm3D = act_Norm3D(idx,:,:);
    tuning_curve = mean(act_Norm3D,3); tuning_curve(prod(tuning_curve,2) == 0,:) = []; 
    array_align = zeros(size(tuning_curve,1),2*numCLS_view-1);
    for ii = 1:size(tuning_curve,1)
        [~,indMax] = max(tuning_curve(ii,:),[],2);
        switch indMax
            case 1
                array_align(ii,[5:9]) = array_align(ii,[5:9])+tuning_curve(ii,1:numCLS_view);
                array_align(ii,[5:-1:1]) = array_align(ii,[5:-1:1])+tuning_curve(ii,1:numCLS_view);
                array_align(ii,5) = array_align(ii,5)/2;
            case 2
                array_align(ii,[4:8]) = array_align(ii,[4:8])+tuning_curve(ii,1:numCLS_view);
                array_align(ii,[6:-1:2]) = array_align(ii,[6:-1:2])+tuning_curve(ii,1:numCLS_view);
                array_align(ii,[4:6]) = array_align(ii,[4:6])/2;
            case 3
                array_align(ii,[3:7]) = array_align(ii,[3:7])+tuning_curve(ii,1:numCLS_view);
                array_align(ii,[7:-1:3]) = array_align(ii,[7:-1:3])+tuning_curve(ii,1:numCLS_view);
                array_align(ii,[3:7]) = array_align(ii,[3:7])/2;
            case 4
                array_align(ii,[2:6]) = array_align(ii,[2:6])+tuning_curve(ii,1:numCLS_view);
                array_align(ii,[8:-1:4]) = array_align(ii,[8:-1:4])+tuning_curve(ii,1:numCLS_view);
                array_align(ii,[4:6]) = array_align(ii,[4:6])/2;
            case 5
                array_align(ii,[1:5]) = array_align(ii,[1:5])+tuning_curve(ii,1:numCLS_view);
                array_align(ii,[9:-1:5]) = array_align(ii,[9:-1:5])+tuning_curve(ii,1:numCLS_view);
                array_align(ii,5) = array_align(ii,5)/2;
        end
    end
    Cell_align_tuning{ll} = array_align./(numCLS_view*size(array_align,2));
end
   
subplot(4,8,[17,18,25,26]); hold on
s1 = plot([-180:45:180],nanmean(Cell_align_tuning{1},1)+(1-mean(nanmean(Cell_align_tuning{1},1))),'k');
shadedErrorBar([-180:45:180],nanmean(Cell_align_tuning{1},1)+(1-mean(nanmean(Cell_align_tuning{1},1))),nanstd(Cell_align_tuning{1},1),'lineprops',{'k','markerfacecolor','k'});
s2 = plot([-180:45:180],nanmean(Cell_align_tuning{2},1)+(1-mean(nanmean(Cell_align_tuning{2},1))),'b');
shadedErrorBar([-180:45:180],nanmean(Cell_align_tuning{2},1)+(1-mean(nanmean(Cell_align_tuning{2},1))),nanstd(Cell_align_tuning{1},1),'lineprops',{'b','markerfacecolor','b'});
s3 = plot([-180:45:180],nanmean(Cell_align_tuning{3},1)+(1-mean(nanmean(Cell_align_tuning{3},1))),'r');
shadedErrorBar([-180:45:180],nanmean(Cell_align_tuning{3},1)+(1-mean(nanmean(Cell_align_tuning{3},1))),nanstd(Cell_align_tuning{1},1),'lineprops',{'r','markerfacecolor','r'});
line([-180 180],[1 1],'Color','k','LineStyle','--')
xlim([-180 180]); xticks([-180:45:180]); ylim([0.5 1.5]);
xlabel('Viewpoint difference (deg)'); ylabel('Normalized response (A.U.)'); legend([s1,s2,s3],'Conv3','Conv4','Conv5','Location','northeast'); title('Untrained network (Fig 5f)');

% Viewpoint invariance (Fig 5g)
array_vi_mean = zeros(3,1); array_vi_std = zeros(3,1);
for ll = 1:3
    vstat = Cell_view_stat{ll+2}; tmp_vi = 1./vstat(:,1); tmp_vi(isinf(tmp_vi)) = []; tmp_vi(isnan(tmp_vi)) = [];
    [tmp_sort,~] = sort(tmp_vi);
    q1 = tmp_sort(floor(0.25*numel(tmp_vi)));
    q3 = tmp_sort(floor(0.75*numel(tmp_vi)));
    iqr = q3-q1;
    idx_outlier = find((tmp_vi>(q3+1.5*iqr))|(tmp_vi<(q1-1.5*iqr)));
    tmp_vi = setdiff(tmp_vi,tmp_vi(idx_outlier));
    array_vi_mean(ll) = mean(tmp_vi); array_vi_std(ll) = std(tmp_vi);
end

subplot(4,8,[20,21,28,29]); hold on
for ll = 1:3
    errorbar([ll+2],array_vi_mean(ll),array_vi_std(ll),'k')
end
plot([3:5],array_vi_mean,'-ok')
xticks([3:5]); xlim([2.5 5.5]); ylim([0 2])
xticklabels({'Conv3','Conv4','Conv5'}); xlabel('Network hierarchy'); ylabel('Invariance index'); title('Viewpoint invariance (Fig 5g)')

% Number of invariant units (Fig 5h)
array_ratio_inv = zeros(3,1);
for ll = 1:3
    array_ratio_inv(ll) = length(sel_cellview_mat{ll+2,1})/length(Cell_Idx{1,1,1,ll+2});
end
subplot(4,8,[23,24,31,32]); hold on
plot([3:5],array_ratio_inv(:)-array_ratio_inv(3),'-ok')
xticks([3:5]); xlim([2.5 5.5]); ylim([min(array_ratio_inv(:)-array_ratio_inv(3))-0.1 max(array_ratio_inv(:)-array_ratio_inv(3))+0.1])
xticklabels({'Conv3','Conv4','Conv5'}); xlabel('Network hierarchy'); ylabel('Ratio change of viewpoint-invariant units'); title('Number of units (Fig 5h)')