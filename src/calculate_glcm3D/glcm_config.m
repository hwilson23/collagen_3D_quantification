% glcm_config.m — thin MATLAB wrapper around config.json.
%
% All actual values live in config.json at the project root.
% Edit that file, not this one.
%
% Loaded automatically by each glcm3D_*.m script via:
%   run(fullfile(fileparts(mfilename('fullpath')), 'glcm_config.m'))

scriptDir   = fileparts(mfilename('fullpath'));          % .../src/calculate_glcm3D
srcDir      = fileparts(scriptDir);                     % .../src
projectRoot = fileparts(srcDir);                        % .../collagen_3D_multimetric-main

raw = jsondecode(fileread(fullfile(projectRoot, 'config.json')));

CFG = raw.glcm;