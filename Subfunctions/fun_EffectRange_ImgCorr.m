function [corr_mat,corr_cont_mat,ER_ImgCorr] = fun_EffectRange_ImgCorr(IMG_var,cls_idx)

class_idx = cls_idx == 1;
IMG_cls = squeeze(IMG_var(:,:,1,class_idx));
corr_mat = zeros(200,13);
corr_cont_mat = zeros(200,13);
for ii = 1:200
    IMG_cen = IMG_cls(:,:,(ii-1)*13+7);
    
    IMG_cont = squeeze(IMG_var(:,:,1,cls_idx ==(randi(4)+1)));
    for vv = 1:13
        IMG_temp = IMG_cls(:,:,(ii-1)*13+vv);
        corr_temp = corrcoef(IMG_cen,IMG_temp);
        corr_mat(ii,vv) = corr_temp(1,2);
        
        IMG_temp = IMG_cont(:,:,(randi(200)-1)*13+vv);
        corr_temp = corrcoef(IMG_cen,IMG_temp);
        corr_cont_mat(ii,vv) = corr_temp(1,2);
    end
end

selective_bool = zeros(1,13);
for vv = 1:13
    if (sum(isnan(corr_mat(:,vv))) == 200)||(sum(isnan(corr_cont_mat(:,vv))) == 200)
        bb = 0;
    else
        [~,bb] = ttest2(corr_mat(:,vv),corr_cont_mat(:,vv));
    end
    bb(isnan(bb)) = 0;
    selective_bool(vv) = bb && (mean(corr_mat(:,vv)) > mean(corr_cont_mat(:,vv)));
end

bw_sel_bool = bwlabel(selective_bool);
max_ii = max(bw_sel_bool);
max_range_mat = zeros(max_ii,1);
for ii = 1:max_ii
    max_range_mat(ii) = sum(selective_bool(selective_bool==ii));
end

ER_ImgCorr = max(max_range_mat);
end

