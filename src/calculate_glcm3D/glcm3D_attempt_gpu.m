%code derived from
%https://github.com/Pedram-Parnianpour/VGLCM-TOP-3D-Texture-Analysis/tree/main 
%https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0117759#sec003

%user input — edit glcm_config.m to change paths or parameters

gpu = gpuDevice(1);
disp(gpu.Name);
%reset(gpu); optional, but would clear anything running

if isempty(gcp('nocreate'))
    parpool('local');
end

run(fullfile(fileparts(mfilename('fullpath')), 'glcm_config.m'));
OPT.D              = CFG.D;
OPT.NeighborSize   = CFG.NeighborSize;
OPT.quantLevel     = CFG.quantLevel;
OPT.glcm_properties = CFG.glcm_properties;
folder  = CFG.folder_gpu;
outpath = CFG.outpath_gpu;
cd(folder)
%load file

files = dir(fullfile(folder, '*.tif'));
filenames = {files.name};
%disp({files.name})
%mask file
fprintf('Starting Time: %s\n',datestr(now));
for i=1:length(filenames)
    timerStart = tic;
    imgpg1 = imread(string(fullfile(folder,filenames(i))));
    disp(filenames(i))
    [numRows, numCols, ~] = size(imgpg1);

    % Get the total number of pages
    info = imfinfo(string(fullfile(folder,filenames(i))));
    numPages = length(info);

    % Preallocate the 3D volume
    volimg = zeros(numRows, numCols, numPages, class(imgpg1));

    % Read each page and store it in the 3D volume
    for page = 1:numPages
        volimg(:, :, page) = imread(string(fullfile(folder,filenames(i))), page);
    end

    volimg = double(volimg);
    volimg = gpuArray(volimg);
    disp(size(volimg))

    %createmask (assuming mask preapplied to image)
    mask = volimg;
    mask(mask>0) = 1;
    mask = logical(mask);

    [d1,d2,d3] = ind2sub(size(mask), find(mask));
    InsideRange = [min(d1) max(d1);min(d2) max(d2);min(d3) max(d3)];
    origsize = size(volimg);
    partialmask = mask(InsideRange(1,1):InsideRange(1,2),InsideRange(2,1):InsideRange(2,2),InsideRange(3,1):InsideRange(3,2));
    volimg = volimg(InsideRange(1,1):InsideRange(1,2),InsideRange(2,1):InsideRange(2,2),InsideRange(3,1):InsideRange(3,2));
    %disp(unique(volimg));
    [~,justname,~] = fileparts(filenames(i));
    glcmFile = fullfile(outpath,string([char(justname) '_3D_D' num2str(OPT.D) '_N' num2str(OPT.NeighborSize) '_Q' num2str(OPT.quantLevel) '.mat'])); 
    disp('gclmfile')
    disp(glcmFile)
    
    if (exist(string(glcmFile), 'file') == 2)
        GLCMS = load(glcmFile);
        GLCMS = gpuArray(GLCMS.GLCMS);
    else
        GLCMS = CreateGLCM_Local(volimg,OPT.quantLevel,[min(volimg(:)) max(volimg(:))],OPT.D,OPT.NeighborSize,partialmask); 
        m = matfile(glcmFile,'writable', true);
        m.GLCMS = gather(GLCMS);
    end
    GLCMS_cpu = gather(GLCMS);
    partialmask_cpu = gather(partialmask);
    texture = computeGLCMLocalFeat(GLCMS_cpu,partialmask_cpu,OPT.glcm_properties);        
    time_total = toc(timerStart);
    fprintf('Total time: %2.2f\n',time_total);
     
    for prop=1:length(OPT.glcm_properties)
        img = gather(squeeze(texture(:,:,:,prop))); 
        img = (img-min(img(:))); %shift to not negative
        img = img/max(img(:))*65535;
        %disp(unique(img))
        %disp(unique(img))
        fullimg = zeros(origsize);
        fullimg(InsideRange(1,1):InsideRange(1,2),InsideRange(2,1):InsideRange(2,2),InsideRange(3,1):InsideRange(3,2)) = img;
        %fullimg(double(mask)==0) = 0;
        %fullimg = uint16(fullimg); %scaling to make sure small values not wiped
        TextureFileName  = [justname '_' OPT.glcm_properties{prop} '_3D_D' num2str(OPT.D) '_N' num2str(OPT.NeighborSize) '_Q' num2str(OPT.quantLevel) '.tif'];
        for z = 1:length(img(1,1,:))
            slice = fullimg(:,:,z);
            if z ==1
                imwrite(uint16(slice),fullfile(outpath,join(TextureFileName)));
            else
                imwrite(uint16(slice),fullfile(outpath,join(TextureFileName)),'WriteMode','append')
            
            end
        end
        
        
    end
    disp("files saved!")
    
end

disp("saving files");
delete(gcp('nocreate'));
fprintf('End Time: %s\n',datestr(now));



function offset = ComputeOffsets(dis)
    [delta1,delta2,delta3] = ndgrid(-dis:dis,-dis:dis,-dis:dis);
    inside = round(sqrt(delta1.*delta1+delta2.*delta2+delta3.*delta3))<=dis;
    offset = [delta1(inside(:)),delta2(inside(:)),delta3(inside(:))];
    offset(offset(:,1)==0&offset(:,2)==0&offset(:,3)==0,:) = [];
    % For symmetric glcm we need half of offsets
    % offset = offset(1:size(offset,1)/2,:);
end

function texture = computeGLCMLocalFeat(glcm,mask,GLCM_feat)
texture = zeros(size(glcm,1),size(glcm,2),size(glcm,3),length(GLCM_feat),'single');
fprintf('Computing Features: ')
[vi, vj, vk] = ind2sub(size(mask),find(mask));
nVoxels = length(vi);
featresults = zeros(nVoxels, length(GLCM_feat),'double');

parfor v = 1:nVoxels
    i = vi(v);
    j = vj(v);
    k = vk(v);

    glcm_norm = squeeze(glcm(i,j,k,:,:));
    featresults(v,:) = computeFeature(glcm_norm, GLCM_feat);
end
for v = 1:nVoxels
    texture(vi(v),vj(v),vk(v),:) = featresults(v,:);
end


%%old version for gpu/regular cpu converting - without allocation for
%%parfor
%{
for i=1:size(glcm,1)
    for j=1:size(glcm,2)
        for k=1:size(glcm,3)
            if(mask(i,j,k)==0)
                continue;
            end
            glcm_norm = gpuArray(squeeze(glcm(i,j,k,:,:)));            
            st = computeFeature(glcm_norm,GLCM_feat); 
            texture(i,j,k,:) = st;
            
        end
    end
    fprintf(repmat('\b',1,n));
    msg = sprintf('%2.2f',i/size(glcm,1)*100);
    disp([msg '%%']);
    n=numel(msg)+1;                    
end
%}
texture = gather(texture);
disp("done with compute fetures")
end

function feature = computeFeature(glcm,GLCM_feat_all)
    feature = zeros(length(GLCM_feat_all),1);
    size_glcm_1 = size(glcm,1); 
    size_glcm_2 = size(glcm,2); 
    Pij = glcm;
    [i,j] = meshgrid(1:size_glcm_1,1:size_glcm_2); 
    p_x = squeeze(sum(Pij,2)); 
    p_y = squeeze(sum(Pij,1))'; 
    glcm_mean = mean(Pij(:)); 
    idx1 = (i+j)-1; 
    p_xplusy = zeros((2*size_glcm_1 - 1),1);
    for aux = 1:max(idx1(:)) 
        p_xplusy(aux) = sum(Pij(idx1==aux)); 
    end 
    ii = (1:(2*size_glcm_1-1))'; 
    jj = (0:size_glcm_1-1)';
    idx2 = abs(i-j)+1; 
    p_xminusy = zeros((size_glcm_1),1);
    for aux = 1:max(idx2(:)) 
        p_xminusy(aux) = sum(Pij(idx2==aux)); 
    end 
    u_x = sum(sum(i.*Pij)); 
    u_y = sum(sum(j.*Pij));     
    s_x = sum(sum(Pij.*((i-u_x).^2)))^0.5; 
    s_y = sum(sum(Pij.*((j-u_y).^2)))^0.5; 
    
    for prop=1:length(GLCM_feat_all)
        GLCM_feat = GLCM_feat_all{prop};
        % Autocorrelation
        if(strcmp(GLCM_feat,'autoc')==1)    
            feature(prop) = sum(sum(Pij.*(i.*j))); 
        % Contrast 
        elseif(strcmp(GLCM_feat,'contr')==1)
            feature(prop) = sum(sum((abs(i-j).^2).*Pij)); 
        % Dissimilarity 
        elseif(strcmp(GLCM_feat,'dissi')==1)
            feature(prop) = sum(sum(abs(i-j).*Pij)); 
        % Energy 
        elseif(strcmp(GLCM_feat,'energ')==1)
            feature(prop) = sum(sum(Pij.^2)); 
        % Entropy 
        elseif(strcmp(GLCM_feat,'entro')==1)
            feature(prop) = -sum(sum(Pij.*log(Pij+eps))); 
        % Homogeneity Matlab 
        elseif(strcmp(GLCM_feat,'homom')==1)
            feature(prop) =  sum(sum(Pij./(1+abs(i-j)))); 
        % Homogeneity Paper 
        elseif(strcmp(GLCM_feat,'homop')==1)
            feature(prop) = sum(sum(Pij./(1+abs(i-j).^2))); 
        % Sum of squares: Variance 
        elseif(strcmp(GLCM_feat,'sosvh')==1)
            feature(prop) = sum(sum(Pij.*((j-glcm_mean).^2))); 
        % Inverse difference normalized 
        elseif(strcmp(GLCM_feat,'indnc')==1)
            feature(prop) = sum(sum(Pij./(1+(abs(i-j)./size_glcm_1)))); 
        % Inverse difference moment normalized 
        elseif(strcmp(GLCM_feat,'idmnc')==1)
            feature(prop) = sum(sum(Pij./(1+((i-j)./size_glcm_1).^2))); 
        % Maximum probability 
        elseif(strcmp(GLCM_feat,'maxpr')==1)
            feature(prop) = max(Pij(:)); 
        % Sum average 
        elseif(strcmp(GLCM_feat,'savgh')==1)
            feature(prop) = sum((ii+1).*p_xplusy); 
        % Sum entropy 
        elseif(strcmp(GLCM_feat,'senth')==1)
            feature(prop) = -sum(p_xplusy.*log(p_xplusy+eps));     
        % Sum variance 
        elseif(strcmp(GLCM_feat,'svarh')==1)
            senth = -sum(p_xplusy.*log(p_xplusy+eps)); 
            feature(prop) = sum((((ii+1) - senth).^2).*p_xplusy); 
        % Difference entropy 
        elseif(strcmp(GLCM_feat,'denth')==1)
            feature(prop) = -sum(p_xminusy.*log(p_xminusy+eps)); 
        % Difference variance 
        elseif(strcmp(GLCM_feat,'dvarh')==1)
            feature(prop) = sum((jj.^2).*p_xminusy); 
        % Correlation Matlab     
        elseif(strcmp(GLCM_feat,'corrm')==1)
            corm = sum(sum(Pij.*(i-u_x).*(j-u_y))); 
            feature(prop) = corm/(s_x*s_y); 
        % Correlation paper 
        elseif(strcmp(GLCM_feat,'corrp')==1)
            corp = sum(sum(Pij.*(i.*j))); 
            feature(prop) = (corp-u_x*u_y)/(s_x*s_y); 
        % Cluster Prominence 
        elseif(strcmp(GLCM_feat,'cprom')==1)
            feature(prop) = sum(sum(Pij.*((i+j-u_x-u_y).^4))); 
        % Cluster Shade    
        elseif(strcmp(GLCM_feat,'cshad')==1)
            feature(prop) = sum(sum(Pij.*((i+j-u_x-u_y).^3)));   
        % Information measure of correlation 1 
        elseif(strcmp(GLCM_feat,'inf1h')==1)
            hx = -sum(p_x.*log(p_x+eps)); 
            hy = -sum(p_y.*log(p_y+eps)); 
            hxy = -sum(sum(Pij.*log(Pij+eps))); 
            hxy1 = -sum(sum(Pij.*log(p_x*p_y' + eps))); 
            feature(prop) = (hxy-hxy1)/(max([hx,hy])); 
        % Information measure of correlation 2
        elseif(strcmp(GLCM_feat,'inf2h')==1)
            hxy = -sum(sum(Pij.*log(Pij+eps))); 
            hxy2 = -sum(sum((p_x*p_y').*log(p_x*p_y' + eps))); 
            feature(prop) = (1-exp(-2*(hxy2-hxy)))^0.5;     
        end  
    end
    feature(isnan(feature)) = 0;
    feature(isinf(feature)) = 0;
    feature = gather(feature);
end

%--------------------------------------------------------------------------
function [GLCMS] = CreateGLCM_Local(I, NL, GL,D,NeighborSizeRad,mask)

I = gpuArray(double(I));
mask = gpuArray(mask);

if GL(2) == GL(1)
    SI = ones(size(I), 'gpuArray');
else
    slope = (NL-1) / (GL(2) - GL(1));
    intercept = 1 - (slope*(GL(1)));
    SI = round(imlincomb(slope,I,intercept,'double'));
end
SI(SI > NL) = NL;
SI(SI < 1) = 1;
vol_Gray = single(SI);
SIZ = gather(size(I));
clear I SI;

[OffsetsMatrixS, OffsetsMatrixE] = AllOffsetsAllNeighbors(vol_Gray,D,NeighborSizeRad,mask);
InsideMask = OffsetsMatrixS>0&OffsetsMatrixE>0;
disp("InsideMask");
disp(size(InsideMask));

glcm = zeros(length(InsideMask),NL,NL,'single', 'gpuArray');
fprintf('Computing GLCM: ')
n=0;
for i=1:NL
    for j=1:NL
        G = zeros(size(InsideMask),'uint8', 'gpuArray');
        G(InsideMask) = vol_Gray(OffsetsMatrixS(InsideMask))==i&vol_Gray(OffsetsMatrixE(InsideMask))==j;
        glcm(:,i,j) = sum(sum(G,3),2);
    end
    fprintf('%s', repmat(sprintf('\b'), 1, n));
    msg = sprintf('%2.2f',i/NL*100);
    fprintf([msg '%%']);
    n=numel(msg)+1;                    
end
fprintf('\n');
clear vol_Gray;
norm_fact = sum(sum(glcm,3),2);

glcm = gather(glcm);
mask = gather(mask);
glcm = glcm./repmat(norm_fact,[1 NL NL]);
mask = repmat(mask,[1 1 1 NL NL]);
GLCMS = zeros(SIZ(1),SIZ(2),SIZ(3),NL,NL,'single');
GLCMS(mask) = glcm;
end

function [OffsetsMatrixS, OffsetsMatrixE] = AllOffsetsAllNeighbors(I,D,NeighborSizeRad,mask)
Y = gpuArray(single(1:size(I,1)));
X = gpuArray(single(1:size(I,2)));
Z = gpuArray(single(1:size(I,3)));
[X,Y,Z] = meshgrid(X,Y,Z);

X = X(mask(:));
Y = Y(mask(:));
Z = Z(mask(:));
IND_s = single(sub2ind(size(I),Y,X,Z));
% Computing all offsets at the voxel
Offsets = gpuArray(single(ComputeOffsets(D)));
% For speed we can use half of offsets
% Offsets = Offsets(1:size(Offsets,1)/2,:);
Neighbors = gpuArray(single(ComputeOffsets(NeighborSizeRad)));
OffsetsMatrixS = single(repmat(IND_s,[1 ,size(Offsets,1),size(Neighbors,1)+1]));
OffsetsMatrixE = zeros(size(OffsetsMatrixS),'single','gpuArray');
fprintf('Computing Maps: ');
n=0;
for ofst=1:size(Offsets,1)
    offset = Offsets(ofst,:);
    Y_e = Y+offset(1);
    X_e = X+offset(2);
    Z_e = Z+offset(3);
    isInMask = Y_e>0&X_e>0&Z_e>0&Y_e<=size(I,1)&X_e<=size(I,2)&Z_e<=size(I,3);
    IND_e = sub2ind(size(I),Y_e(isInMask),X_e(isInMask),Z_e(isInMask));%End point of vector
    %Check if it is masked
    isInMask2 = mask(IND_e);
    isInMask(isInMask) = isInMask2;
    OffsetsMatrixE(isInMask,ofst,1) = IND_e(isInMask2);
    % Computing all offsets at the neighbors of the voxel
    for nb=1:size(Neighbors,1)
        NeighborOffset = Neighbors(nb,:);
        Y_ns = Y+NeighborOffset(1);
        X_ns = X+NeighborOffset(2);
        Z_ns = Z+NeighborOffset(3);
        isInMask = Y_ns>0&X_ns>0&Z_ns>0&Y_ns<=size(I,1)&X_ns<=size(I,2)&Z_ns<=size(I,3);
        IND_ns = sub2ind(size(I),Y_ns(isInMask),X_ns(isInMask),Z_ns(isInMask));
        isInMask2 = mask(IND_ns);
        isInMask(isInMask) = isInMask2;
        OffsetsMatrixS(isInMask,ofst,nb+1) = IND_ns(isInMask2);
        %Check if masked
        Y_ne = Y_e+NeighborOffset(1);
        X_ne = X_e+NeighborOffset(2);
        Z_ne = Z_e+NeighborOffset(3);
        isInMask = Y_ne>0&X_ne>0&Z_ne>0&Y_ne<=size(I,1)&X_ne<=size(I,2)&Z_ne<=size(I,3);
        IND_ne = sub2ind(size(I),Y_ne(isInMask),X_ne(isInMask),Z_ne(isInMask));
        isInMask2 = mask(IND_ne);
        isInMask(isInMask) = isInMask2;
        OffsetsMatrixE(isInMask,ofst,nb+1) = IND_ne(isInMask2);
    end
    fprintf(repmat('\b',1,n));
    msg = sprintf('%2.2f',ofst/size(Offsets,1)*100);
    fprintf([msg '%%']);
    n=numel(msg)+1;                
end
fprintf('\n');
end