function [array_SVM_face_multi,array_SVM_face_multi_shuf] = fun_SVM(net_rand,num_cell,IND_Face,IMG_cell,idx_mat,layersSet,indLayer,numNeuron)

numIMGtot = 120; idx_class = [1,2,3,5,6,7];
IMG = zeros(227,227,3,numIMGtot);
for cc = 1:6
    IMG_mat = IMG_cell{idx_class(cc),1}(:,:,idx_mat{idx_class(cc),2});
    IMG_mat = repmat(permute(IMG_mat,[1 2 4 3]),[1 1 3 1]);
    switch cc
        case 1
            idx_IMG = 1:numIMGtot/2;
            IMG(:,:,:,idx_IMG) = IMG_mat(:,:,:,randperm(size(IMG_mat,4),numIMGtot/2));
        otherwise
            idx_IMG = numIMGtot/(2*5)*(cc-2)+1:numIMGtot/(2*5)*(cc-1); idx_IMG = idx_IMG+numIMGtot/2;
            IMG(:,:,:,idx_IMG) = IMG_mat(:,:,:,randperm(size(IMG_mat,4),numIMGtot/(2*5)));
    end
end
IMG = single(IMG);

%% SVM
reN = 1;
ratio_Tr_Te = 3;
numFace_ratio = 1;
array_ratio = [numNeuron];

% multiple neuron (305 neurons)
array_SVM_face_multi = zeros(reN,numFace_ratio);
array_SVM_face_multi_shuf = zeros(reN,numFace_ratio);
% split train / test
iFace = [1:numIMGtot/2]; iObject = [numIMGtot/2+1:numIMGtot]; iObject = iObject(randperm(length(iObject)));
iFaceTrain = iFace(randperm(numIMGtot/2,numIMGtot/2/ratio_Tr_Te*2)); iFaceTest = setdiff(iFace,iFaceTrain);
iObjectTrain = iObject(randperm(numIMGtot/2,numIMGtot/2/ratio_Tr_Te*2)); iObjectTest = setdiff(iObject,iObjectTrain);

indTrain = [iFaceTrain';iObjectTrain']; indTest = [iFaceTest';iObjectTest'];
YTrain = [ones(numIMGtot/2/ratio_Tr_Te*2,1); zeros(numIMGtot/2/ratio_Tr_Te*2,1)];
YTest = [ones(numIMGtot/2/ratio_Tr_Te,1); zeros(numIMGtot/2/ratio_Tr_Te,1)];

% response
act_rand = activations(net_rand,IMG,layersSet{indLayer});
act = reshape(act_rand,num_cell,size(IMG,4));

XTrain1 = act(IND_Face,indTrain); XTest1 = act(IND_Face,indTest);
sXTrain1 = reshape(XTrain1(randperm(length(IND_Face)*length(indTrain))),[length(IND_Face),length(indTrain)]);
%% Multiple neuron SVM
for ii = 1:numFace_ratio
    tmpN = array_ratio(ii);
%     disp([' # neuron =',num2str(tmpN)])
    for rr = 1:reN
        ind_f = randperm(length(IND_Face),tmpN);
        Mdl = fitcecoc(XTrain1(ind_f,:)',YTrain);
        YPredict = predict(Mdl,XTest1(ind_f,:)');
        array_SVM_face_multi(rr,ii) = length(find(YTest == YPredict))./length(YTest);
        
        Mdl = fitcecoc(sXTrain1(ind_f,:)',YTrain);
        YPredict = predict(Mdl,XTest1(ind_f,:)');
        array_SVM_face_multi_shuf(rr,ii) = length(find(YTest == YPredict))./length(YTest);
    end
end
end