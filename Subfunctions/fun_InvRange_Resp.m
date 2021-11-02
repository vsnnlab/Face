function [resp_z_mat,face_resp_z_mat,max_resp_z_mat] = fun_InvRange_Resp(act_face,num_face_cell,cls_idx,var_idx)

resp_z_mat = zeros(num_face_cell,13,5,200);
face_resp_z_mat = zeros(num_face_cell,13,200);
max_resp_z_mat = zeros(num_face_cell,13,200);

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
end

end

