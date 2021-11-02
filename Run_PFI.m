%% Preferred feature images of face-selective units in untrained networks (Fig.2, Fig.S4) 

%% Preferred feature images (PFI) (Fig 2c)
Cell_PFI = cell(numCLS-1,2);
if Sim == 0                                                                % 1 : Fast version of PFI simulation. The saved PFI would be displayed.
    load('Data_PFI_XDream_ClsUnit.mat','PFI_XDream_mat')
    load('Data_PFI_RevCorr_ClsUnit.mat','PFI_RevCorr_mat')

    for cc = 1:numCLS-1
        for ii = 1:2
            switch ii
                case 1
                    Cell_PFI{cc,1} = PFI_XDream_mat{cc};
                case 2
                    Cell_PFI{cc,2} = PFI_RevCorr_mat{cc};
            end
        end
    end
elseif Sim == 1                                                            % 2 : Actual calculation process would be run. It takes around 30 minutes.
    net_rand = Cell_Net{1,1,1}; IND_face = Cell_Idx{1,1,1,length(layerArray)};
        
    %% Reverse correlation (Fig 2a)
    % Stimulus parameters
    N_image = 2500;         % Number of stimulus images
    iteration = 100;         % Number of iteration
    img_size = 227;         % Image size
    dot_size = 5;           % Size of 2D Gaussian filter
    
    % Generate 2D Gaussian filters
    [pos_xx,pos_yy] = meshgrid(linspace(1+dot_size,img_size-dot_size,sqrt(N_image)),linspace(1+dot_size,img_size-dot_size,sqrt(N_image)));
    pos_xy_list = pos_xx(:) + 1i*pos_yy(:);
    [xx_field,yy_field] = meshgrid(1:img_size,1:img_size); xy_field = xx_field + 1i*yy_field;
    
    img_list = zeros(img_size,img_size,3,length(pos_xy_list));
    count = 1;
    for pp = 1:length(pos_xy_list)
        pos_tmp = pos_xy_list(pp);
        img_tmp = repmat(exp(-(abs(xy_field-pos_tmp).^2)/2/dot_size.^2)*0.5,1,1,3);
        img_list(:,:,:,count) = -img_tmp;
        count = count + 1;
    end
    Gau_stimulus = cat(4,img_list,-img_list);
    
    % Iterative PFI calculation
    PFI = zeros(img_size,img_size,3)+255/2;                                         %Initial PFI
    PFI_mat = zeros(img_size,img_size,iteration+1); PFI_mat(:,:,1) = PFI(:,:,1);
    
    for iter = 1:iteration
        tic
        PFI_0 = PFI; % Save previous PFI
        
        % Generate stimulus as a summation of previous PFI and gaussian stimulus
        IMG = repmat(PFI/255,[1,1,1,size(Gau_stimulus,4)])+Gau_stimulus;
        IMG = uint8(IMG*255);     IMG(IMG<0) = 0; IMG(IMG>255) = 255;
        
        % Measure the response of random AlexNet
        act_rand = activations(net_rand,IMG,'relu5');       % Response of 'relu5' layer in random AlexNet
        act_reshape = reshape(act_rand,43264,size(IMG,4));  % Reshape the response in 2D form
        act_reshape_sel = act_reshape(IND_face,:);          % Find the response of face-selective neurons
        mean_act = mean(act_reshape_sel,1);                 % Average response of face-selective neurons
        
        % Calculate the PFI
        norm_act_reshape = repmat(permute(mean_act-min(mean_act),[1,3,4,2]),img_size,img_size,3);
        PFI = sum(norm_act_reshape.*double(IMG),4)/sum(mean_act-min(mean_act));
        PFI_diff = PFI-PFI_0; PFI = PFI_0 + PFI_diff*10;
        PFI(PFI<0) = 0; PFI(PFI>255) = 255;
        PFI_mat(:,:,iter+1) = PFI(:,:,1);
        toc
    end
    Cell_PFI{1,2} = PFI_mat(:,:,end);
    
    %% XDream (Fig 2b)
    % If you want to regenerate XDream PFI, please install the XDream from https://github.com/willwx/XDream
    
    % Alternatively, there is a customized XDream code we wrote. you can also refer to the code below. 
% =====================================================================
%     dlnetGenerator = // pretrained GAN // % you should download or train GAN network
%     numIteration = 101;
%     pElite = 0.01;
%     pHeredity = 0.75;
%     pMutant = 0.01;
%     stdMutant = 0.75;
%     
%     numValid = 10*10; 
%     numLatentInputs = 500;
%     executionEnvironment = "auto";
%     
%     Zvalidation = randn(1,1, numLatentInputs, numValid, 'single');
%     dlZValidation = dlarray(Zvalidation, 'SSCB');
%     
%     if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
%         dlZValidation = gpuArray(dlZValidation);
%     end
%     
%     array_act = zeros(numIteration,numValid);
%     iter_dlZValidation = dlZValidation;
%     Cell_data = cell(numIteration,1);
%     
%     figure; start = tic;
%     for iteration = 1:numIteration
%         tic
%         % generate images
%         dlXGeneratedValidation = predict(dlnetGenerator, iter_dlZValidation);
%         iterIMG = imresize(gather(255.*rescale(extractdata(dlXGeneratedValidation))),[227 227]);
%         
%         % estimate score (face neurons)
%         act_reshape_mat = [];
%         act_rand = activations(net_rand,iterIMG,layersSet{5}); act = reshape(act_rand,num_cell,numValid);
%         actNorm = act;
%         ind_Neuron = IND_face;
%         act_reshape_mat = cat(1,act_reshape_mat,actNorm(ind_Neuron,:));
%         
%         tmpScore = mean(act_reshape_mat,1);
%         
%         array_act(iteration,:) = tmpScore;
%         Cell_data{iteration,1} = iterIMG;
%         
%         disp(['Iteration = ',num2str(iteration-1),' / Score = ',num2str(mean(tmpScore))])
%         if mod(iteration, 1) == 0 || iteration == 1
%             I = imtile(extractdata(dlXGeneratedValidation));
%             I = rescale(I); image(I)
%             
%             D = duration(0,0,toc(start), 'Format', 'hh:mm:ss');
%             title(['Iteration: ', num2str(iteration), ', Elapsed: ', char(D)]);
%             drawnow
%         end
%         
%         % optimize score
%         % current population
%         next_dlZValidation = zeros(1,1,numLatentInputs,numValid,'single');
%         tmp_dlZValidation = gather(extractdata(iter_dlZValidation));
%         
%         % elite population
%         [~,order] = sort(tmpScore,'descend');    % select elite
%         indElite = order(1:floor(pElite*numValid));
%         next_dlZValidation(:,:,:,1:length(indElite)) = tmp_dlZValidation(:,:,:,indElite);
%         
%         % recombination population
%         indRest = setdiff([1:numValid],indElite);
%         
%         w = exp(tmpScore(indRest) - min(tmpScore(indRest)))./std(tmpScore(indRest)); p = w./sum(w);
%         tmpExpectNum = round(length(indRest).*p);
%         ind_list = []; for ii = 1:length(indRest); ind_list = [ind_list; ii.*ones(tmpExpectNum(ii),1)];end % roulette wheel selection
%         
%         new_dlZValidation = zeros(1,1,numLatentInputs,length(indRest),'single');
%         for ii = 1:length(indRest)
%             % cross-over
%             indParent = ind_list(randperm(length(ind_list),2));
%             parants = [squeeze(tmp_dlZValidation(:,:,:,indRest(indParent(1))))...
%                 squeeze(tmp_dlZValidation(:,:,:,indRest(indParent(2))))];
%             
%             tmpPoint = randi(numLatentInputs-floor(numLatentInputs*pHeredity),1);
%             indPre = tmpPoint:tmpPoint+floor(numLatentInputs*pHeredity); indCross = setdiff([1:numLatentInputs],indPre);
%             
%             child = parants(:,1); child(indCross) = parants(indCross,2);
%             % mutation
%             if rand(1)<pMutant
%                 child = child+randn(size(child)).*stdMutant;
%             end
%             new_dlZValidation(1,1,:,ii) = child;
%         end
%         next_dlZValidation(:,:,:,length(indElite)+1:end) = new_dlZValidation;
%         next_dlZValidation = dlarray(next_dlZValidation, 'SSCB');
%         if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
%             next_dlZValidation = gpuArray(next_dlZValidation);
%         end
%         
%         % next generation population
%         iter_dlZValidation = next_dlZValidation;
%         toc
%     end
% =====================================================================
    % 
end

order_ClsIMG = [1 2 4 3 5];
idx_ClsIMG = [56 192 46 86 166];
figure('units','normalized','outerposition',[0 0 0.5 1]); drawnow
sgtitle('Result 2 : Preferred feature images (PFI) (Fig.2, Fig.S4)')
for cc = 1:numCLS-1
    % stimulus
    subplot(numCLS-1,3,(cc-1)*3+1);
    imagesc(IMG_ORI(:,:,1,(order_ClsIMG(cc)-1)*numIMG+idx_ClsIMG(order_ClsIMG(cc)))); colormap(gray); axis image off;
    title(STR_LABEL{order_ClsIMG(cc)})
    
    % PFI (X-Dream)
    subplot(numCLS-1,3,(cc-1)*3+2);
    imshow(mean(Cell_PFI{cc,1},3)); caxis([0 255]); axis image off;
    if cc == 1; title('PFI (X-Dream)'); end
    
    % PFI (Reverse corelation)
    subplot(numCLS-1,3,(cc-1)*3+3);
    imagesc(Cell_PFI{cc,2}(:,:,1)); colormap(gray); caxis([0 255]); axis image off;
    if cc == 1; title('PFI (Reverse-correlation)'); end
end
    
%% t-SNE analysis (Fig S3b-c)
% disp(['Demension reduction analysis ... (~ 1 min)'])
% InitialY = 1e-4*randn(size(IMG_ORI,4),2); Perplexity = 30;
% labels = []; for cc = 1:numCLS-1; labels = [labels;cc.*ones(numIMG,1)];end
% cmap = flip(jet(numCLS+2)); cmap = cmap(round(linspace(1,numCLS+2,numCLS)),:); cmap = cmap(2:end,:); cmap = cmap(order_ClsIMG,:); sz = 10;
% 
% tic
% % Image
% actIMG = reshape(IMG_ORI(:,:,1,:),prod(size(IMG_ORI,1:2)),size(IMG_ORI,4));
% tSNE_IMG = tsne(actIMG','InitialY',InitialY,'Perplexity',Perplexity,'Standardize',0);  
% [~,~,SI_IMG] = fun_IntraInterSI(tSNE_IMG,numCLS-1,numIMG,labels);
% 
% % Conv5 response
% net_rand = Cell_Net{1,1,1};
% num_cell = prod(array_sz(length(layerArray),:));
% act_rand = activations(net_rand,IMG_ORI,layersSet{length(layerArray)});
% act = reshape(act_rand,num_cell,size(IMG_ORI,4));
% tSNE_Resp = tsne(act','InitialY',InitialY,'Perplexity',Perplexity,'Standardize',0);
% [~,~,SI_Resp] = fun_IntraInterSI(tSNE_Resp,numCLS-1,numIMG,labels);
% toc
% 
% figure('units','normalized','outerposition',[0.5 0 0.5 1]); drawnow
% sgtitle('Figure 2 : Demension reduction analysis (Fig S3b-c)')
% subplot(4,4,[1,2,5,6]); hold on
% for ii = numCLS-1:-1:1
%     idx = find(double(labels) == ii);
%     h = gscatter(tSNE_IMG(idx,1),tSNE_IMG(idx,2),labels(idx),cmap(ii,:),'.',sz,'off');
% end
% % colormap(cmap(2:end,:)); cbh = colorbar; cbh.Ticks = 0.1:0.2:0.9 ; cbh.TickLabels = STR_LABEL(1:numCLS); legend off;
% ylabel('tSNE axis 2'); xlim([-50 50]); ylim([-50 50]); title('Raw images (tSNE)');
% 
% subplot(4,4,[3,4,7,8]); hold on
% [sSI,orderSI] = sort(mean(SI_IMG,2),'descend');
% for cc = numCLS-1:-1:1
%     bar([cc],mean(SI_IMG(orderSI(cc),:),2),'facecolor',cmap(orderSI(cc),:))
%     errorbar([cc],mean(SI_IMG(orderSI(cc),:),2),std(SI_IMG(orderSI(cc),:),[],2),'k')
% end
% xticks([1:numCLS-1]); xlim([0.5 numCLS-1+0.5]); xticklabels(STR_LABEL(orderSI));
% ylim([-0.5 0.5]); ylabel(['Silhouette index']); title('Raw images (Silhouette index)');
% 
% 
% subplot(4,4,[9,10,13,14]); hold on
% for ii = numCLS-1:-1:1
%     idx = find(double(labels) == ii);
%     h = gscatter(tSNE_Resp(idx,1),tSNE_Resp(idx,2),labels(idx),cmap(ii,:),'.',sz,'off');
% end
% % colormap(cmap(2:end,:)); cbh = colorbar; cbh.Ticks = 0.1:0.2:0.9 ; cbh.TickLabels = STR_LABEL(1:numCLS); legend off;
% xlabel('tSNE axis 1'); ylabel('tSNE axis 2'); xlim([-50 50]); ylim([-50 50]); title('Conv response in untrained networks (tSNE)');
% 
% subplot(4,4,[11,12,15,16]); hold on
% [sSI,orderSI] = sort(mean(SI_Resp,2),'descend');
% for cc = numCLS-1:-1:1
%     bar([cc],mean(SI_Resp(orderSI(cc),:),2),'facecolor',cmap(orderSI(cc),:))
%     errorbar([cc],mean(SI_Resp(orderSI(cc),:),2),std(SI_Resp(orderSI(cc),:),[],2),'k')
% end
% xticks([1:numCLS-1]); xlim([0.5 numCLS-1+0.5]); xticklabels(STR_LABEL(orderSI));
% ylim([-0.5 0.5]);ylabel(['Silhouette index']); title('Conv response in untrained networks (Silhouette index)');