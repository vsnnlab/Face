function toolbox_chk
% Requires 3 matlab toolboxes ( Released later than 2017) :
% Computer vision system, image processing, parallel computing

warning('on')

matlab_version=ver;
toolbox_names={matlab_version(:).Name};
toolbox_Release={matlab_version(:).Release};

Index_matlab = find(strcmp(toolbox_names,'MATLAB'));
if ~isempty(Index_matlab)
    release_yr_matlab=str2double(regexp(toolbox_Release{Index_matlab},'\d+','match'));
    release_chk_matlab = release_yr_matlab >= 2017;
else
    release_chk_matlab = false;
end

if isempty(Index_matlab) ||  ~release_chk_matlab
    warning('MATLAB must be upgraded to version R2017a or above')
end

Index_vision = find(contains(toolbox_names,'Deep Learning Toolbox'));
if ~isempty(Index_vision)
    release_yr_vision=str2double(regexp(toolbox_Release{Index_vision},'\d+','match'));
    release_chk_vision = release_yr_vision >= 2017;
else
    release_chk_vision = false;
end

if isempty(Index_vision) ||  ~release_chk_vision
    disp('Deep Learning Toolbox must be installed or upgraded to version R2017a or above','r')
    disp('INSTALLATION INSTRUCTIONS')
    disp(' 1. You can download Deep Learning Toolbox at https://www.mathworks.com/products/deep-learning.html')
    disp(' 2. You can download Deep Learning Toolbox in Matlab add-on manager')
    disp('    - Go to the Home tabr')
    disp('    - Click add-on icon')
    disp('    - Search and download Deep Learning Toolbox Model for AlexNet Network')
end