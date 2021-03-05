function [d] = fun_FSI(rep)
% idx_nan = find(isnan(sum(mean(rep,3),2))); rep(idx_nan,:,:) = [];

d = zeros(size(rep,1),1);
for ii = 1:size(d,1)
    [~,indOrder] = sort(mean(rep(ii,:,:),3),'descend');
    d(ii) = (mean(rep(ii,1,:),3)-mean(rep(ii,indOrder(2),:),3))./sqrt((std(rep(ii,1,:),[],3).^2+std(rep(ii,indOrder(2),:),[],3).^2)./2);
end
end