function [resp_obj_z,resp_obj_z_3D] = fun_ResZscore(resp_rand,num_cell,face_idx,numCLS,numIMG)
resp_reshape = reshape(resp_rand,num_cell,numCLS*numIMG);

resp_obj = resp_reshape(face_idx,1:numCLS*numIMG);
resp_obj_3D = zeros(length(face_idx),numCLS,numIMG);
for ii = 1:numCLS
    resp_obj_ii = resp_obj(:,(ii-1)*numIMG+1:numIMG*ii);
    resp_obj_3D(:,ii,:) = resp_obj_ii;
end

resp_obj_z = zeros(length(face_idx),numCLS*numIMG);
resp_obj_z_3D = zeros(length(face_idx),numCLS,numIMG);
resp_obj_max_z = zeros(length(face_idx),numIMG);
for ii = 1:length(face_idx)
    resp_obj_ii = resp_obj(ii,:);
    resp_obj_3D_ii = squeeze(resp_obj_3D(ii,:,:));
    [~,max_obj_idx] = max(mean(resp_obj_3D_ii(2:end,:),2));
    norm_mean = mean(resp_obj_3D_ii(max_obj_idx+1,:));
    norm_std = std(resp_obj_3D_ii(max_obj_idx+1,:));
    resp_obj_z(ii,:) = (resp_obj_ii-norm_mean)/norm_std;
    resp_obj_z_3D(ii,:,:) = (resp_obj_3D_ii-norm_mean)/norm_std;
    resp_obj_max_z(ii,:) = (resp_obj_3D_ii(max_obj_idx+1,:)-norm_mean)/norm_std;
end
end