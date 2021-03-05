function [net_test, lim, Weights, Biases] = fun_Initializeweight(net, ver, stdfac)
net_test = net;
net_tmp = net_test.saveobj;

rand_layers_ind = [2, 6, 10, 12 14];
Weights = cell(1,length(rand_layers_ind));
Biases = cell(1,length(rand_layers_ind));
for ind_tl = 1:length(rand_layers_ind)
    % ind_tl = 1;
    % LOI = layers_set{ind_tl};
    targetlayer_ind = rand_layers_ind(ind_tl);
    weight_conv = net.Layers(targetlayer_ind ,1).Weights;
    bias_conv = net.Layers(targetlayer_ind ,1).Bias;
    
    fan_in = size(weight_conv,1)*size(weight_conv,2)*size(weight_conv,3);
    
    if ver == 1
        lim(ind_tl) = sqrt(2/fan_in);
        Wtmp = stdfac*randn(size(weight_conv))*sqrt(1/fan_in); % LeCun initializaation
        Btmp = randn(size(bias_conv));
    elseif ver == 2
        lim(ind_tl) = sqrt(3/fan_in);
        Wtmp = stdfac*(rand(size(weight_conv))-0.5)*2*sqrt(3/fan_in); % Lecun uniform initializaation
        Btmp = randn(size(bias_conv));
    end

    %% change network parameters
    
    weight_conv_randomize = single(1*Wtmp);
    bias_conv_randomize = single(0*Btmp);
    
    Weights{ind_tl} = weight_conv_randomize;
    Biases{ind_tl} = bias_conv_randomize;
    
    net_tmp.Layers(targetlayer_ind).Weights = weight_conv_randomize;
    net_tmp.Layers(targetlayer_ind).Bias = bias_conv_randomize;
end
net_test = net_test.loadobj(net_tmp);
end