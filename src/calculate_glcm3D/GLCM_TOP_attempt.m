



%code derived from
%https://github.com/Pedram-Parnianpour/VGLCM-TOP-3D-Texture-Analysis/tree/main 
%https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0117759#sec003

%user input — edit glcm_config.m to change paths or parameters

if isempty(gcp('nocreate'))
    parpool('local');
end

run(fullfile(fileparts(mfilename('fullpath')), 'glcm_config.m'));
OPT.D              = CFG.D;
OPT.NeighborSize   = CFG.NeighborSize;
OPT.quantLevel     = CFG.quantLevel;
OPT.glcm_properties = CFG.glcm_properties;
folder  = CFG.folder_ep;
outpath = CFG.outpath_TOP;
cd(folder)
%load file

files = dir(fullfile(folder, '*.tif'));
filenames = {files.name};
%disp({files.name})
%mask file


offset = ComputeOffsets(OPT.D);

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

	texture = zeros(size(volimg,1),size(volimg,2),size(volimg,3),length(OPT.glcm_properties),3,'single');
    timerStart = tic;
    fprintf('================\n');
    
    
       
    for v = 1:3%view
        fprintf('Plane %d: ',v);
        n=0;
        for s=1:size(volimg,v) 
            fprintf(repmat('\b',1,n));
            msg = [num2str(s) '/' num2str(size(volimg,v))];
            fprintf(msg);
            n=numel(msg);            
            [I maskI] = readImgFromVol(volimg,mask,v,s);
            I(~maskI) = NaN;
            [d1,d2] = ind2sub(size(maskI), find(maskI));
            InsideRangeI = [min(d1) max(d1);min(d2) max(d2)];
            if(isempty(InsideRangeI))
                continue;
            end
            
			GLCM = CreateGLCM_Local(I,offset,OPT.quantLevel,[min(volimg(:)) max(volimg(:))],OPT.NeighborSize,InsideRangeI); 
            GLCM = mean(GLCM,5);% Averaging all offsets
            switch(v)
			       
                case 1
                    texture(s,:,:,:,v) = computeGLCMLocalFeat(GLCM,InsideRangeI,OPT.glcm_properties);
                case 2
                    texture(:,s,:,:,v) = computeGLCMLocalFeat(GLCM,InsideRangeI,OPT.glcm_properties);
                case 3
                    texture(:,:,s,:,v) = computeGLCMLocalFeat(GLCM,InsideRangeI,OPT.glcm_properties);
            end
        end
        fprintf('\n');
    end
    % Averaging over views
    texture = mean(texture,5);    
    time_total = toc(timerStart);
    fprintf('Total time: %2.2f\n',time_total);
    
    for prop=1:length(OPT.glcm_properties)
        img = squeeze(texture(:,:,:,prop)); 
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
end
disp("saving files");
fprintf('End Time: %s\n',datestr(now));

%--------------------------------------------------------------------------
function offset = ComputeOffsets(dis)
    [delta1,delta2] = ndgrid(-dis:dis,-dis:dis);
    inside = round(sqrt(delta1.*delta1+delta2.*delta2))<=dis;
    offset = [delta1(inside(:)),delta2(inside(:))];
    offset(offset(:,1)==0&offset(:,2)==0,:) = [];
    % For symmetric glcm we need half of offsets
%     offset = offset(1:size(offset,1)/2,:);
end
%--------------------------------------------------------------------------


function texture = computeGLCMLocalFeat(glcm,InsideRange,GLCM_feat)
texture = zeros(size(glcm,1),size(glcm,2),length(GLCM_feat),'single');
glcm = glcm(InsideRange(1,1):InsideRange(1,2),InsideRange(2,1):InsideRange(2,2),:,:);
for i=1:size(glcm,1)
    for j=1:size(glcm,2)
            g = squeeze(glcm(i,j,:,:));
            st = computeFeature(g,GLCM_feat); 
            texture(i+InsideRange(1,1)-1,j+InsideRange(2,1)-1,:) = st;
    end
end
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
%for aux = 1:max(idx1(:)) 
    %p_xplusy(aux) = sum(Pij(idx1==aux)); 
    p_xplusy = accumarray(idx1(:),Pij(:));
%end 
ii = (1:(2*size_glcm_1-1))'; 
jj = (0:size_glcm_1-1)';
idx2 = abs(i-j)+1; 
p_xminusy = zeros((size_glcm_1),1);
%parfor aux = 1:max(idx2(:)) 
 %   p_xminusy(aux) = sum(Pij(idx2==aux));
     p_xminusy = accumarray(idx2(:),Pij(:));
%end 

u_x = sum(sum(i.*Pij)); 
u_y = sum(sum(j.*Pij));     
s_x = sum(sum(Pij.*((i-u_x).^2)))^0.5; 
s_y = sum(sum(Pij.*((j-u_y).^2)))^0.5; 

for prop=1:length(GLCM_feat_all)
	GLCM_feat = GLCM_feat_all{prop};
	switch GLCM_feat
	% Autocorrelation
		case 'autoc'    
			feature(prop) = sum(sum(Pij.*(i.*j))); 
	% Contrast 
		case'contr'
			feature(prop) = sum(sum((abs(i-j).^2).*Pij)); 
	% Dissimilarity 
		case 'dissi'
			feature(prop) = sum(sum(abs(i-j).*Pij)); 
	% Energy 
		case 'energ' 
			feature(prop) = sum(sum(Pij.*Pij)); 
	% Entropy 
		case 'entro'
			feature(prop) = -sum(sum(Pij.*log(Pij+eps))); 
	% Homogeneity Matlab 
		case 'homom'
			feature(prop) =  sum(sum(Pij./(1+abs(i-j)))); 
	% Homogeneity Paper 
		case 'homop'
			feature(prop) = sum(sum(Pij./(1+abs(i-j).^2))); 
	% Sum of squares: Variance 
		case 'sosvh'
			feature(prop) = sum(sum(Pij.*((j-glcm_mean).*(j-glcm_mean)))); 
	% Inverse difference normalized 
		case 'indnc'
			feature(prop) = sum(sum(Pij./(1+(abs(i-j)./size_glcm_1)))); 
	% Inverse difference moment normalized 
		case 'idmnc'
			feature(prop) = sum(sum(Pij./(1+((i-j)./size_glcm_1).^2))); 
	% Maximum probability 
		case 'maxpr'
			feature(prop) = max(Pij(:)); 
	% Sum average 
		case 'savgh'
			feature(prop) = sum((ii+1).*p_xplusy); 
	% Sum entropy 
		case 'senth'
			feature(prop) = -sum(p_xplusy.*log(p_xplusy+eps));     
	% Sum variance 
		case 'svarh'
			senth = -sum(p_xplusy.*log(p_xplusy+eps)); 
			feature(prop) = sum((((ii+1) - senth).^2).*p_xplusy); 
	% Difference entropy 
		case 'denth'
		feature(prop) = -sum(p_xminusy.*log(p_xminusy+eps)); 
	% Difference variance 
		case 'dvarh'
			feature(prop) = sum((jj.*jj).*p_xminusy); 
	% Correlation Matlab     
		case 'corrm'
			corm = sum(sum(Pij.*(i-u_x).*(j-u_y))); 
			feature(prop) = corm/(s_x*s_y); 
	% Correlation paper 
		case 'corrp'
			corp = sum(sum(Pij.*(i.*j))); 
			feature(prop) = (corp-u_x*u_y)/(s_x*s_y); 
	% Cluster Prominence 
		case 'cprom' 
			feature(prop) = sum(sum(Pij.*((i+j-u_x-u_y).^4))); 
	% Cluster Shade    
		case 'cshad'
			feature(prop) = sum(sum(Pij.*((i+j-u_x-u_y).^3)));   
	% Information measure of correlation 1 
		case 'inf1h'
			hx = -sum(p_x.*log(p_x+eps)); 
			hy = -sum(p_y.*log(p_y+eps)); 
			hxy = -sum(sum(Pij.*log(Pij+eps))); 
			hxy1 = -sum(sum(Pij.*log(p_x*p_y' + eps))); 
			feature(prop) = (hxy-hxy1)/(max([hx,hy])); 
	% Information measure of correlation 2
		case 'inf2h'
			hxy = -sum(sum(Pij.*log(Pij+eps))); 
			hxy2 = -sum(sum((p_x*p_y').*log(p_x*p_y' + eps))); 
			feature(prop) = (1-exp(-2*(hxy2-hxy)))^0.5;     
	end  
end
feature(isnan(feature)) = 0;
feature(isinf(feature)) = 0;
end

%--------------------------------------------------------------------------

function [I maskI]=readImgFromVol(volimg,mask,view,sliceNO)
switch(view)
    case 1
        I = squeeze(volimg(sliceNO,:,:));
        maskI = squeeze(mask(sliceNO,:,:));
    case 2
        I = squeeze(volimg(:,sliceNO,:));
        maskI = squeeze(mask(:,sliceNO,:));
    case 3
        I = squeeze(volimg(:,:,sliceNO));
        maskI = squeeze(mask(:,:,sliceNO));
end
end
%--------------------------------------------------------------------------
function [GLCMS] = CreateGLCM_Local(I, Offset, NL, GL,NeighborSize,InsideRange)
Iorig = I;
I = I(InsideRange(1,1):InsideRange(1,2),InsideRange(2,1):InsideRange(2,2));
if GL(2) == GL(1)
    SI = ones(size(I));
else
    slope = (NL-1) / (GL(2) - GL(1));
    intercept = 1 - (slope*(GL(1)));
    SI = round(imlincomb(slope,I,intercept,'double'));
end
SI(SI > NL) = NL;
SI(SI < 1) = 1;
numOffsets = size(Offset,1);

if NL ~= 0
    s = size(I);
    [r,c] = meshgrid(1:s(1),1:s(2));
    r = r(:);
    c = c(:);
    % Compute GLCMS
    GLCMS = zeros(size(Iorig,1),size(Iorig,2),NL,NL,numOffsets,'single');
    for k = 1 : numOffsets
%         fprintf('========%d=======',k);
        GLCMS(InsideRange(1,1):InsideRange(1,2),InsideRange(2,1):InsideRange(2,2),:,:,k) = computeGLCM(r,c,Offset(k,:),SI,NL,NeighborSize);
%         squeeze(GLCMS(InsideRange(1,1)+55,InsideRange(2,1)+15,:,:,k))
    end
else
	GLCMS = zeros(0,0,numOffsets);
end

end
%--------------------------------------------------------------------------
function [oneGLCM] = computeGLCM(r,c,offset,si,nl,NeighborSize)
oneGLCM = zeros(size(si,1),size(si,2),nl,nl);
Neigh = ones(NeighborSize*2+1);
r2 = r + offset(1);
c2 = c + offset(2);
[nRow nCol] = size(si);
outsideBounds = find(c2 < 1 | c2 > nCol | r2 < 1 | r2 > nRow);
Index1 = r + (c - 1)*nRow;
Index1(outsideBounds) = [];
r2(outsideBounds) = []; 
c2(outsideBounds) = [];
Index2 = r2 + (c2 - 1)*nRow;
hasNoValue = isnan(si(Index1)) | isnan(si(Index2));
Index1 = Index1(~hasNoValue);
Index2 = Index2(~hasNoValue);
for i=1:nl
    for j=1:nl
        hasVec = zeros(size(si));
        Index = Index1(si(Index1)==i&si(Index2)==j);
        if(isempty(Index))
            oneGLCM(:,:,i,j) = zeros(size(si));
        else
            hasVec(Index) = 1;
            oneGLCM(:,:,i,j) = conv2(hasVec,Neigh,'same');
        end
    end
end
end
%--------------------------------------------------------------------------
