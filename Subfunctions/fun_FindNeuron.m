function [cell_idx] = fun_FindNeuron(act_rand,num_cell,numCLS,numIMG,pThr,indClass)

act_reshape = reshape(act_rand,num_cell,numCLS*numIMG);
act_3D = zeros(num_cell,numCLS,numIMG);

for cc = 1:numCLS
    act_3D(:,cc,:) = act_reshape(:,(cc-1)*numIMG+1:cc*numIMG);
end

pref_class = zeros(num_cell,1);
sel_p_val = zeros(num_cell,1);

for cc = 1:num_cell
    mean_FR = [];
    for cls = 1:numCLS
        mean_FR = [mean_FR,mean(act_reshape(cc,(cls-1)*numIMG+1:cls*numIMG))];
    end
    
    [~,sort_ind] = sort(mean_FR,'descend');
    pref_class(cc) = sort_ind(1);
    
    resp1 = act_reshape(cc,(sort_ind(1)-1)*numIMG+1:numIMG*sort_ind(1));
    pval_temp = [];
    for ee = 2:numCLS
        resp2 = act_reshape(cc,(sort_ind(ee)-1)*numIMG+1:numIMG*sort_ind(ee));
        pval_temp(ee-1) = ranksum(resp1,resp2);
    end
    sel_p_val(cc) = max(pval_temp);
end

for ii = indClass
    cell_idx = find((pref_class==ii) & (sel_p_val<pThr));
end

end