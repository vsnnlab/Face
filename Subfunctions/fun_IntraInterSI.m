function [array_Intra,array_Inter,array_Sindex] = fun_IntraInterSI(Y,numCLS,numIMG,labels)
array_Intra = zeros(numCLS,numIMG);
array_Inter = zeros(numCLS,numIMG);
array_Sindex = zeros(numCLS,numIMG);

for cc = 1:numCLS
    restClass = setdiff([1:numCLS],cc);
    tmpintra = find(labels == cc);
    for ii = 1:numIMG
        intra = setdiff(tmpintra,ii);
        array_Intra(cc,ii) = mean(sqrt((Y(intra,1)-Y(ii,1)).^2+(Y(intra,2)-Y(ii,2)).^2));
        %% raw image
        interD = [];
        for ccc = 1:length(restClass)
            inter = find(double(labels) == restClass(ccc));
            interD = [interD; mean(sqrt((Y(inter,1)-Y(ii,1)).^2+(Y(inter,2)-Y(ii,2)).^2))];
        end
        array_Inter(cc,ii) = min(interD);
        array_Sindex(cc,ii) = (array_Inter(cc,ii)-array_Intra(cc,ii))./max(array_Intra(cc,ii),array_Inter(cc,ii));
    end
end
end