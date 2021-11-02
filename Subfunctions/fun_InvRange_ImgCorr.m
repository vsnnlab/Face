function [corr_mat,corr_cont_mat] = fun_InvRange_ImgCorr(IMG_var,cls_idx)

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

end

