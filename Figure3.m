%% Figure 3 : Detection of face images using the response of face units in untrained networks 

%% Train SVM using response of an untrained network (Fig 3b-d)
net_rand = Cell_Net{1,1,1};                                                % untrained AlexNet                                                    
num_cell = prod(array_sz(layerArray(length(layerArray)),:));
load('IMG_obj_201201.mat','IMG_cell','idx_mat')

%% Find neuron
disp('Prepare SVM training for face detection task ... (~ 40 sec)')
tic
act_rand = activations(net_rand,IMG_ORI,layersSet{length(layerArray)});
actORIre = reshape(act_rand,num_cell,size(IMG_ORI,4));

Idx_All = [1:num_cell]';
Idx_Face = Cell_Idx{1,1,1,length(layerArray)};

p = zeros(num_cell,1);
for mm = 1:num_cell
    meanRF =[mean(actORIre(mm,1:numIMG)),mean(actORIre(mm,numIMG+1:2*numIMG)),mean(actORIre(mm,2*numIMG+1:3*numIMG)),...
        mean(actORIre(mm,3*numIMG+1:4*numIMG)),mean(actORIre(mm,4*numIMG+1:5*numIMG)),mean(actORIre(mm,5*numIMG+1:6*numIMG))];
    if sum(meanRF) == 0
        p(mm) = 1;
    else
        [~,cls_max] = max(meanRF); [~,cls_min] = min(meanRF);
        p(mm) = ranksum(actORIre(mm,numIMG*(cls_max-1)+1:numIMG*cls_max),actORIre(mm,numIMG*(cls_min-1)+1:numIMG*cls_min));
    end
end
arrayClass = [ones(numIMG,1);2.*ones(numIMG,1);3.*ones(numIMG,1);4.*ones(numIMG,1);5.*ones(numIMG,1);6.*ones(numIMG,1)];
p2 = zeros(num_cell,1); for mm = 1:num_cell;p2(mm) = anova1(actORIre(mm,:),arrayClass,'off');end
Idx_NS = intersect(find(p>0.9),find(p>0.9));
toc

%% SVM train using response of single unit (Fig 3b)
reN = 5;                                                                   % number of repetition for SVM training, N = 100 in the manuscript
num_Sample_singe = 20;                                                     % number of randomly sampled units, all units was used in the manuscript
Cell_SVM_single = cell(3,1);                                               % 1: Face-selective; 2: Response shuffled; 3: Non-selective

% Face-unit
disp('Train SVM using response of singe face unit... (~ 1 min 30 sec)')
cell_list = Idx_Face(randperm(length(Idx_Face),num_Sample_singe));
array_SVM = zeros(length(cell_list),reN);  
array_SVM_shuf = zeros(length(cell_list),reN);
tic
for ii = 1:length(cell_list)
    if mod(ii,10) == 0; disp(['% ',num2str(ii),' / ',num2str(length(cell_list))]); end
    for rr = 1:reN
        [array_SVM(ii,rr),array_SVM_shuf(ii,rr)] = fun_SVM(net_rand,num_cell,cell_list(ii),IMG_cell,idx_mat,layersSet,length(layerArray),length(cell_list(ii)));
    end
end
toc
Cell_SVM_single{1} = array_SVM; Cell_SVM_single{3} = array_SVM_shuf;

% Non-selective 
disp('Train SVM using response of singe non-selective unit... (~ 1 min 30 sec)')
cell_list = Idx_NS(randperm(length(Idx_NS),num_Sample_singe));
array_SVM = zeros(length(cell_list),reN);  
array_SVM_shuf = zeros(length(cell_list),reN);
tic
for ii = 1:length(cell_list)
    if mod(ii,10) == 0; disp(['% ',num2str(ii),' / ',num2str(length(cell_list))]); end
    for rr = 1:reN
        [array_SVM(ii,rr),~] = fun_SVM(net_rand,num_cell,cell_list(ii),IMG_cell,idx_mat,layersSet,length(layerArray),length(cell_list(ii)));
    end
end
toc
Cell_SVM_single{2} = array_SVM;

%% SVM train using response of multiple units (Fig 3c,d)
reN = 10;                                                                  % number of repetition for SVM training, N = 100 in the manuscript
numFace_ratio = 10;                                                        
array_ratio = round(logspace(log10(1),log10(length(Idx_Face)),numFace_ratio)); 
array_ratio(end) = length(Idx_Face);
Cell_SVM_mult = cell(3,1);                                                 % 1: Face-selective; 2: Non-selective; 3: Conv5

% Face-unit
disp('Train SVM using response of multi face units... (~ 1 min 30 sec)')
tic
cell_list = Idx_Face;
array_SVM = zeros(length(array_ratio),reN); 
for vv = 1:length(array_ratio)
    if mod(vv,5) == 0; disp(['% ',num2str(vv),' / ',num2str(length(array_ratio))]); end
    for ii = 1:reN
        [array_SVM(vv,ii),~] = fun_SVM(net_rand,num_cell,cell_list,IMG_cell,idx_mat,layersSet,length(layerArray),array_ratio(vv));
    end
end
Cell_SVM_mult{1} = array_SVM;
toc 


% Non-selective 
disp('Train SVM using response of multi non-selective units... (~ 1 min 30 sec)')
tic
cell_list = Idx_NS;
array_SVM = zeros(length(array_ratio),reN); 
for vv = 1:length(array_ratio)
    if mod(vv,5) == 0; disp(['% ',num2str(vv),' / ',num2str(length(array_ratio))]); end
    for ii = 1:reN
        [array_SVM(vv,ii),~] = fun_SVM(net_rand,num_cell,cell_list,IMG_cell,idx_mat,layersSet,length(layerArray),array_ratio(vv));
    end
end
Cell_SVM_mult{2} = array_SVM;
toc 


% Conv5 
disp('Train SVM using response all Conv5 units... (~ 30 sec)')
tic
cell_list = Idx_All;
array_SVM = zeros(1,reN);
for ii = 1:reN
    [array_SVM(1,ii),~] = fun_SVM(net_rand,num_cell,cell_list,IMG_cell,idx_mat,layersSet,length(layerArray),length(cell_list));
end
Cell_SVM_mult{3} = array_SVM;
toc
clearvars IMG_cell idx_mat


figure('units','normalized','outerposition',[0 0.5 1 0.5]); drawnow
sgtitle('Figure 3 : Face detection task using the response of face units (Fig 3b-d)')
tmpColor_SVM = [1 0 0;1 0 0;0.7 0.7 0.7]; 
subplot(2,6,[1,2,7,8]); hold on
for ii = 1:3
    bar([ii],mean(mean(Cell_SVM_single{ii},2),1),0.5,'facecolor',tmpColor_SVM(ii,:))
    errorbar([ii],mean(mean(Cell_SVM_single{ii},2),1),std(mean(Cell_SVM_single{ii},2),1),'k')
end
xticks([1:3]); xlim([0.5 3.5]); ylim([0.4 1.1])
xticklabels({['Face-selective\newline{         unit}'];['Response\newline{  shuffled}'];['Non-selective\newline{       unit}']}); ylabel('Correct ratio');
title('Single-unit performance (Fig 3b)')

subplot(2,6,[3,4,9,10]); hold on
shadedErrorBar(array_ratio,mean(Cell_SVM_mult{1},2),std(Cell_SVM_mult{1},[],2),'lineprops','r');
s1 = plot(array_ratio,mean(Cell_SVM_mult{1},2),'r');
shadedErrorBar(array_ratio,mean(Cell_SVM_mult{2},2),std(Cell_SVM_mult{2},[],2),'lineprops','k');
s2 = plot(array_ratio,mean(Cell_SVM_mult{2},2),'k');
s3 = line([1 length(Idx_Face)],[mean(Cell_SVM_mult{3},2) mean(Cell_SVM_mult{3},2)],'Color','k','LineStyle','--');
set(gca, 'XScale', 'log'); xlim([1 length(Idx_Face)]); ylim([0.4 1.1])  
xlabel('Number of units'); ylabel('Correct ratio'); 
legend([s1,s2,s3],'Face-selective unit','Non-selective unit','All units (n=43,264)','Location','northwest')
title('Multi-unit performance (Fig 3c)')

subplot(2,6,[5,6,11,12]); hold on
bar([1],mean(Cell_SVM_mult{3},2),0.5,'facecolor','k')
errorbar([1],mean(Cell_SVM_mult{3},2),std(Cell_SVM_mult{3},[],2),'k')
bar([2],mean(Cell_SVM_mult{1}(end,:),2),0.5,'facecolor','r')
errorbar([2],mean(Cell_SVM_mult{1}(end,:),2),std(Cell_SVM_mult{1}(end,:),[],2),'k')
xticks([1:2]); xlim([0.5 2.5]); ylim([0.4 1.1])
xticklabels({['All\newline{unit}'];['Face-selective\newline{         unit}']}); ylabel('Correct ratio');
title('Multi-unit performance (Fig 3d)')