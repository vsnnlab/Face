%% Invariant characteristics of face-selective units in untrained networks (Fig.S5,7) 

%% Analysis for invariance to image variation (Fig.S5a-f)
net_rand = Cell_Net{1,1,1};                                                % untrained AlexNet  
Idx_Face = Cell_Idx{1,1,1,length(layerArray)};                             % indices of face units in the untrained AlexNet
num_cell = prod(array_sz(layerArray(length(layerArray)),:));

Cell_var_axis = cell(TT,1);
Cell_resp = cell(TT,1);
Cell_face_resp = cell(TT,1);
Cell_nonface_resp = cell(TT,1);

StrTitle = {'Translation','Scaling','Rotation'};
StrUnit = {'r_R_F','%','deg'};
arrayXlim = [-1.5, 0, -200;1.5, 250, 200];
StrXlabel = {'Translation (r_R_F)','Face size change (%)','Rotation (deg)'};
arrayYlim = [3, 300, 400];
StrYlabel = {'Effective range (r_R_F)','Effective range (%)','Effective range (deg)'};

for vtype = 1:TT
    tic
    %% Load feature variant stimulus set
    switch vtype
        case 1
            disp('Position')
            load('IMG_var_pos_210521.mat'); IMG_var = single(repmat(permute(IMG_pos,[1 2 4 3]),[1 1 3])); clearvars IMG_pos
            var_idx = pos_idx; clearvars pos_idx
            RF_size = 163/2;
            var_axis = (-120:20:120)/RF_size; 
        case 2
            disp('Size')
            load('IMG_var_size_210521.mat'); IMG_var = single(repmat(permute(IMG_size,[1 2 4 3]),[1 1 3])); clearvars IMG_size
            var_idx = size_idx; clearvars size_idx
            RF_size = 163;
            var_axis = (41:25:341)/RF_size*100;
        case 3
            disp('Rotation')
            load('IMG_var_rot_210521.mat'); IMG_var = single(repmat(permute(IMG_rot,[1 2 4 3]),[1 1 3])); clearvars IMG_rot
            var_idx = rot_idx; clearvars rot_idx
            RF_size = 1;
            var_axis = -180:30:180;
    end
    Cell_var_axis{vtype} = var_axis;
    
    %% Measure network response 
    act_rand = activations(net_rand,IMG_var,layersSet{layerArray(length(layerArray))});
    act_re = reshape(act_rand,num_cell,size(IMG_var,4));
    act_face = act_re(Idx_Face,:);
    num_face_cell = size(Idx_Face,1);
    clearvars act_rand act_re
    
    %% Measure effective range
    [resp_z_mat,face_resp_z_mat,max_resp_z_mat] = fun_InvRange_Resp(act_face,num_face_cell,cls_idx,var_idx);
    
    Cell_resp{vtype} = resp_z_mat;
    Cell_face_resp{vtype} = face_resp_z_mat;
    Cell_nonface_resp{vtype} = max_resp_z_mat;
     
    %% Plot figure for each image variation
    figure('units','normalized','outerposition',[0 0 1 1]); drawnow
    sgtitle(['Invariant charateristics of face units (',StrTitle{vtype},') (Fig.S5)'])
    
    % stimulus (Fig.S5a)
    pos_subplot = [1:5]; pos_img = linspace(1,13,5); img_idx = find(cls_idx == 1);
    for ii = 1:5
        subplot(4,5,pos_subplot(ii)); imagesc(IMG_var(:,:,1,img_idx(pos_img(ii)))); axis image off; colormap(gray);
    end
    pos_subplot = [6:10]; pos_img = linspace(1,13,5); img_idx = find(cls_idx == 5);
    for ii = 1:5
        subplot(4,5,pos_subplot(ii)); imagesc(IMG_var(:,:,1,img_idx(pos_img(ii)))); axis image off; colormap(gray);
        switch vtype
            case 1
                title([num2str(round(Cell_var_axis{vtype}(pos_img(ii)),1)),' ',StrUnit{vtype}])
            otherwise
                title([num2str(round(Cell_var_axis{vtype}(pos_img(ii)),0)),' ',StrUnit{vtype}])
        end
    end
    
    % face tuning curve (Fig.S5b)
    resp_cat_mat = squeeze(nanmean(Cell_resp{vtype},1));
    pos_img = [7,10,13];
    subplot(4,5,[11 12 16 17]); hold on
    s1 = plot([0 1 2 3 4 5 6],[0 mean(resp_cat_mat(:,:,pos_img(1))) 0],'color',[1 0 0]);
    s2 = plot([0 1 2 3 4 5 6],[0 mean(resp_cat_mat(:,:,pos_img(2))) 0],'color',[1 0 1]);
    s3 = plot([0 1 2 3 4 5 6],[0 mean(resp_cat_mat(:,:,pos_img(3))) 0],'color',[0 0 1]);
    xticks([1:numCLS-1]); xlim([0.5 numCLS-1+0.5]); xticklabels(STR_LABEL); 
    Str_leg = cell(1,3);
    switch vtype
        case 1
            for ii = 1:3
                Str_leg{ii} = [num2str(round(Cell_var_axis{vtype}(pos_img(ii)),1)),' ',StrUnit{vtype}];
            end
        otherwise
            for ii = 1:3
                Str_leg{ii} = [num2str(round(Cell_var_axis{vtype}(pos_img(ii)),0)),' ',StrUnit{vtype}];
            end
    end
    legend([s1,s2,s3],Str_leg,'Location','southeast'); ylabel('Response (z-scored)');
    title('Face tuning curve (Fig.S5b)')
    
    % Response of face units to image variation (Fig.S5c-d)
    subplot(4,5,[14 15 19 20]); hold on
    face_resp_z_mat = Cell_face_resp{vtype};
    max_resp_z_mat = Cell_nonface_resp{vtype};
    
    face_resp_z_mat(face_resp_z_mat==inf) = nan;
    face_resp_z_mean = squeeze(nanmean(face_resp_z_mat,1));
    max_resp_z_mean = squeeze(nanmean(max_resp_z_mat,1));
    
    shadedErrorBar(Cell_var_axis{vtype},nanmean(face_resp_z_mean,2),nanstd(face_resp_z_mean,[],2)./sqrt(10),'lineprops','r');
    shadedErrorBar(Cell_var_axis{vtype},nanmean(max_resp_z_mean,2),nanstd(max_resp_z_mean,[],2)./sqrt(10),'lineprops','k');
    s1 = plot(Cell_var_axis{vtype},nanmean(face_resp_z_mean,2),'color',[1 0 0]);
    s2 = plot(Cell_var_axis{vtype},nanmean(max_resp_z_mean,2),'color',[0 0 0]);
    xlim([arrayXlim(1,vtype),arrayXlim(2,vtype)]); ylim([-0.4 1.2]); xlabel(StrXlabel{vtype}); ylabel('Response (z-scored)')
    title('Response of face units (Fig.S5c,d)'); legend([s1,s2],{'Face stimulus','Non-face stimulus'},'Location','northeast')

    clearvars IMG_cls IMG_var
    toc
end