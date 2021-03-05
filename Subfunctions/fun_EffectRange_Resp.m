function [resp_z_mat,face_resp_z_mat,max_resp_z_mat,ER_single_mat] = fun_EffectRange_Resp(act_face,num_face_cell,cls_idx,var_idx)

resp_z_mat = zeros(num_face_cell,13,5,200);
face_resp_z_mat = zeros(num_face_cell,13,200);
max_resp_z_mat = zeros(num_face_cell,13,200);
ER_single_mat = zeros(num_face_cell,1);

for nn = 1:num_face_cell
    for vv = 1:13
        resp_temp = zeros(5,200);
        for cc = 1:5
            idx_tmp = (cls_idx == cc) & (var_idx == vv);
            resp_temp(cc,:) = act_face(nn,idx_tmp);
        end
        
        [~,max_idx] = max(mean(resp_temp(2:5,:),2));
        max_mean = mean(resp_temp(max_idx+1,:));
        max_std = std(resp_temp(max_idx+1,:));
        
        temp = (resp_temp(1,:)-max_mean)/max_std;
        temp(abs(temp) == inf) = nan;
        face_resp_z_mat(nn,vv,:) = temp;
        
        temp = (resp_temp(max_idx+1,:)-max_mean)/max_std;
        temp(abs(temp) == inf) = nan;
        max_resp_z_mat(nn,vv,:) = temp;
        
        temp = (resp_temp-max_mean)/max_std;
        temp(abs(temp) == inf) = nan;
        resp_z_mat(nn,vv,:,:) = temp;
    end
    
    face_tuning_single = squeeze(face_resp_z_mat(nn,:,:));
    max_tuning_single = squeeze(max_resp_z_mat(nn,:,:));
    selective_bool = zeros(1,13);
    for vv = 1:13
        if (sum(isnan(face_tuning_single(vv,:))) == 200)||(sum(isnan(max_tuning_single(vv,:))) == 200)
            bb = 0;
        else
            [~,bb] = ttest2(face_tuning_single(vv,:),max_tuning_single(vv,:));
        end
        bb(isnan(bb)) = 0;
        selective_bool(vv) = bb && (mean(face_tuning_single(vv,:)) > mean(max_tuning_single(vv,:)));
    end
    bw_sel_bool = bwlabel(selective_bool);
    max_ii = max(bw_sel_bool);
    max_range_mat = zeros(max_ii,1);
    for ii = 1:max_ii
        max_range_mat(ii) = sum(selective_bool(bw_sel_bool==ii));
    end
    if isempty(max_range_mat)
        max_range_mat = 0;
    end
    
    ER_single_mat(nn) = max(max_range_mat);
end

end

