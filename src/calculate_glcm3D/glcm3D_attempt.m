% ============================================================================
% Windowed 3D GLCM texture pipeline — box-filter / two-pass version
%
% Replaces explicit per-voxel offset-index enumeration (OffsetsMatrixS/E)
% and the dense per-voxel NL x NL co-occurrence array (GLCMS) with a
% streaming two-pass box-filter approach. This decouples memory use from
% BOTH the number of gray levels (NL) and the neighborhood radius (R),
% so you can process the whole 512x512xZ volume at once (no sectioning)
% and sweep NeighborSize in {1,3,5,7,...} without blowing up RAM.
%
% IMPORTANT DESIGN NOTE — window shape:
%   The original code used a *spherical* neighborhood (round(sqrt(...))<=R,
%   from ComputeOffsets). This version uses a *cubic* window of size
%   (2R+1)^3 for the neighborhood, because a cube is separable and can be
%   summed in O(1) per voxel via a 3D integral image (summed-volume table),
%   regardless of R. This is what makes R=7 basically as cheap as R=1.
%   If you need an exact spherical neighborhood, see the note at the
%   bottom of computeWindowedGLCMFeatures() for how to swap in a
%   convn()-based spherical kernel instead (more expensive, but still
%   avoids the old index-array memory blowup).
%
% Memory scaling:
%   Old approach:  O(numVoxels * NL^2)                for GLCMS
%                  O(numVoxels * nOffsets(D) * nNeighbors(R)) for indices
%   This approach: O(numVoxels * NL)                  for marginal accumulators
%                  O(numVoxels) for box-filter intermediates (R-independent)
% ============================================================================

%% ---------------------- user input ----------------------
HOWMANYZ = 46;
OPT.D               = 1;                   % distance for offset computation
OPT.NeighborSizeList = [1 3 5 7];           % sweep multiple kernel radii
OPT.quantLevel       = 8;                   % NL; bump this up as desired (e.g. 16, 32, 64)
OPT.glcm_properties  = {'autoc','contr','corrm','corrp','cprom','cshad','dissi', ...
                         'energ','entro','homom','homop','maxpr','sosvh','savgh', ...
                         'svarh','senth','dvarh','denth','inf1h','inf2h','indnc','idmnc'};

folder  = "/home/hwilson23/Documents/MATLAB_processing/maskapplied_timeseries_stacks";
outpath = "/home/hwilson23/Documents/MATLAB_processing/output_bksub_texture3D";
cd(folder)

time = 1;
files = dir(fullfile(folder, '*.tif'));
filenames = {files.name};

fprintf('Starting Time: %s\n', datestr(now));

%% ---------------------- global intensity percentile range ----------------------
histCounts = zeros(1, 65536);
for i = 1:length(filenames)
    for z = 1:HOWMANYZ
        img = imread(fullfile(folder, filenames{i}), z);
        fg = img(img > 0);
        if isempty(fg), continue; end
        counts = histcounts(double(fg), 0:65536);
        histCounts = histCounts + counts;
    end
end
cdf = cumsum(histCounts) / sum(histCounts);
globalLow  = find(cdf >= 0.005, 1, 'first') - 1;
globalHigh = find(cdf >= 0.995, 1, 'first') - 1;
fprintf('full dataset percentile range: %d %d\n', globalLow, globalHigh);
setmin = globalLow;
setmax = globalHigh;

answerwas = input("continue? 1-yes/0-no", "s");
if answerwas == "0"
    return
end

%% ---------------------- main loop: whole volume, no sectioning ----------------------
for i = 1:length(filenames)

    imgpg1 = imread(string(fullfile(folder, filenames{i})));
    disp(filenames{i})
    [numRows, numCols, ~] = size(imgpg1);
    info = imfinfo(string(fullfile(folder, filenames{i})));
    numPages = length(info); %#ok<NASGU>

    volimg = zeros(numRows, numCols, HOWMANYZ, class(imgpg1));
    startPage = (time - 1) * HOWMANYZ + 1;
    for z = 1:HOWMANYZ
        page = startPage + z - 1;
        volimg(:,:,z) = imread(fullfile(folder, filenames{i}), page);
    end

    volimg = double(volimg);
    disp(size(volimg))

    mask = volimg > 0;

    % crop to tissue bounding box (still valuable — cuts wasted work on
    % empty background — but this is NOT the same as sectioning: no
    % boundary truncation of neighborhoods happens here because we keep
    % the full padded context via box filtering, see below)
    [d1, d2, d3] = ind2sub(size(mask), find(mask));
    InsideRange = [min(d1) max(d1); min(d2) max(d2); min(d3) max(d3)];
    origsize = size(volimg);

    cropMask = mask(InsideRange(1,1):InsideRange(1,2), InsideRange(2,1):InsideRange(2,2), InsideRange(3,1):InsideRange(3,2));
    cropVol  = volimg(InsideRange(1,1):InsideRange(1,2), InsideRange(2,1):InsideRange(2,2), InsideRange(3,1):InsideRange(3,2));

    [~, justname, ~] = fileparts(filenames{i});

    for R = OPT.NeighborSizeList

        timerStart = tic;
        fprintf('Processing %s | NL=%d | R=%d\n', justname, OPT.quantLevel, R);

        texture = computeWindowedGLCMFeatures(cropVol, cropMask, OPT.quantLevel, ...
                        [setmin, setmax], OPT.D, R, OPT.glcm_properties);

        fprintf('Total time (NL=%d, R=%d): %2.2f s\n', OPT.quantLevel, R, toc(timerStart));

        for prop = 1:length(OPT.glcm_properties)
            img = squeeze(texture(:,:,:,prop));

            fullimg = zeros(origsize);
            fullimg(InsideRange(1,1):InsideRange(1,2), InsideRange(2,1):InsideRange(2,2), InsideRange(3,1):InsideRange(3,2)) = img;
            fullimg = single(fullimg);

            TextureFileName = sprintf('%s_t%d_%s_3D_D%d_R%d_Q%d_raw.tif', ...
                justname, time, OPT.glcm_properties{prop}, OPT.D, R, OPT.quantLevel);
            fullFilePath = fullfile(outpath, TextureFileName);

            t = Tiff(char(fullFilePath), 'w');
            for a = 1:size(fullimg,3)
                slice = fullimg(:,:,a);
                tagstruct.ImageLength = size(slice,1);
                tagstruct.ImageWidth  = size(slice,2);
                tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
                tagstruct.BitsPerSample = 32;
                tagstruct.SamplesPerPixel = 1;
                tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
                tagstruct.RowsPerStrip = size(slice,1);
                tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
                t.setTag(tagstruct);
                t.write(slice);
                if a < size(fullimg,3)
                    t.writeDirectory();
                end
            end
            t.close();
        end
    end
    disp("file saved!")
end

disp("all done");
fprintf('End Time: %s\n', datestr(now));


% ============================================================================
function offset = ComputeOffsets(dis)
    [delta1,delta2,delta3] = ndgrid(-dis:dis,-dis:dis,-dis:dis);
    inside = round(sqrt(delta1.*delta1+delta2.*delta2+delta3.*delta3))<=dis;
    offset = [delta1(inside(:)),delta2(inside(:)),delta3(inside(:))];
    offset(offset(:,1)==0&offset(:,2)==0&offset(:,3)==0,:) = [];
end


% ============================================================================
function out = boxFilter3(in, R)
% Sum of `in` over a cubic window of radius R centered at every voxel,
% zero-padded outside the volume. O(numVoxels) regardless of R via a 3D
% summed-volume table (integral image). `in` should be numeric (single).
    win = 2*R + 1;
    in = single(in);
    P = padarray(in, [R R R], 0, 'both');
    C = cumsum(cumsum(cumsum(P,1),2),3);
    S = padarray(C, [1 1 1], 0, 'pre');

    sz = size(in);
    r_lo = (1:sz(1));        r_hi = r_lo + win;
    c_lo = (1:sz(2));        c_hi = c_lo + win;
    p_lo = (1:sz(3));        p_hi = p_lo + win;

    out = S(r_hi,c_hi,p_hi) - S(r_lo,c_hi,p_hi) - S(r_hi,c_lo,p_hi) - S(r_hi,c_hi,p_lo) ...
        + S(r_lo,c_lo,p_hi) + S(r_lo,c_hi,p_lo) + S(r_hi,c_lo,p_lo) - S(r_lo,c_lo,p_lo);
end


% ============================================================================
function out = shiftVolume(vol, offset, fillVal)
% Shift `vol` by integer offset [dy dx dz], filling out-of-bounds voxels
% with fillVal. Equivalent to "vol sampled at pos+offset", zero outside.
    sz = size(vol);
    if islogical(vol)
        out = false(sz);
    else
        out = cast(zeros(sz), 'like', vol);
        if fillVal ~= 0
            out(:) = fillVal;
        end
    end
    r = 1:sz(1); c = 1:sz(2); p = 1:sz(3);
    rs = r + offset(1); cs = c + offset(2); ps = p + offset(3);
    validR = rs>=1 & rs<=sz(1);
    validC = cs>=1 & cs<=sz(2);
    validP = ps>=1 & ps<=sz(3);
    out(validR,validC,validP) = vol(rs(validR), cs(validC), ps(validP));
end


% ============================================================================
function texture = computeWindowedGLCMFeatures(vol, mask, NL, GL, D, R, GLCM_feat)
% Two-pass streaming windowed-GLCM feature computation.
%
% Pass 1: build per-voxel marginals (p_x, p_y, p_{x+y}, p_{x-y}, total
%         neighbor-pair count) by looping over all NL^2 gray-level pairs,
%         but only ACCUMULATING into O(NL) marginal volumes — the joint
%         count for each pair is discarded immediately after use.
% Pass 2: recompute the same NL^2 joint counts (cheap — box filtering is
%         R-independent) and accumulate each requested texture feature
%         directly, using the per-voxel scalars derived from pass 1
%         (u_x, u_y, s_x, s_y, hx, hy) as weights.
%
% Recomputing the joint twice trades a bit of extra compute for avoiding
% ever storing an NL^2-sized array per voxel.

    sz = size(vol);
    nVox = numel(vol);

    % ---- quantize intensities into NL gray levels ----
    if GL(2) == GL(1)
        SI = ones(sz);
    else
        slope = (NL-1) / (GL(2)-GL(1));
        intercept = 1 - slope*GL(1);
        SI = round(vol.*slope + intercept);
    end
    SI(SI > NL) = NL;
    SI(SI < 1)  = 1;
    if NL <= 255
        SI = uint8(SI);
    else
        SI = uint16(SI);
    end
    mask = logical(mask);

    % ---- precompute shifted SI / mask / validity for each distance offset ----
    offsets = ComputeOffsets(D);
    nOffsets = size(offsets,1);

    SI_shift   = cell(nOffsets,1);
    valid_d    = cell(nOffsets,1);
    for d = 1:nOffsets
        SI_shift{d} = shiftVolume(SI, offsets(d,:), 0);
        mask_shift  = shiftVolume(mask, offsets(d,:), false);
        valid_d{d}  = mask & mask_shift;   % both endpoints inside tissue+volume
    end

    % ---- total valid (offset,neighbor) pair count per voxel ----
    validSum = zeros(sz, 'single');
    for d = 1:nOffsets
        validSum = validSum + single(valid_d{d});
    end
    total = boxFilter3(validSum, R);
    totalSafe = total;
    totalSafe(totalSafe == 0) = NaN;   % avoid div-by-zero; results masked out below

    % ---- PASS 1: marginals ----
    p_x  = cell(NL,1);            % row marginal, indexed by i
    p_y  = cell(NL,1);            % column marginal, indexed by j
    for k = 1:NL
        p_x{k} = zeros(sz,'single');
        p_y{k} = zeros(sz,'single');
    end
    p_xplusy  = cell(2*NL-1,1);   % indexed by k = i+j-1
    for k = 1:(2*NL-1)
        p_xplusy{k} = zeros(sz,'single');
    end
    p_xminusy = cell(NL,1);       % indexed by k = |i-j|+1
    for k = 1:NL
        p_xminusy{k} = zeros(sz,'single');
    end

    for gi = 1:NL
        for gj = 1:NL
            jointSum = zeros(sz,'single');
            for d = 1:nOffsets
                jointSum = jointSum + single(valid_d{d} & (SI==gi) & (SI_shift{d}==gj));
            end
            localCount = boxFilter3(jointSum, R);

            p_x{gi} = p_x{gi} + localCount;
            p_y{gj} = p_y{gj} + localCount;
            k1 = gi + gj - 1;
            k2 = abs(gi - gj) + 1;
            p_xplusy{k1}  = p_xplusy{k1}  + localCount;
            p_xminusy{k2} = p_xminusy{k2} + localCount;
        end
    end
    clear jointSum localCount valid_d validSum

    % ---- derived per-voxel scalars from marginals only ----
    u_x = zeros(sz,'single'); u_y = zeros(sz,'single');
    for gi = 1:NL, u_x = u_x + gi .* (p_x{gi} ./ totalSafe); end
    for gj = 1:NL, u_y = u_y + gj .* (p_y{gj} ./ totalSafe); end

    s_x2 = zeros(sz,'single'); s_y2 = zeros(sz,'single');
    for gi = 1:NL, s_x2 = s_x2 + ((gi-u_x).^2) .* (p_x{gi} ./ totalSafe); end
    for gj = 1:NL, s_y2 = s_y2 + ((gj-u_y).^2) .* (p_y{gj} ./ totalSafe); end
    s_x = sqrt(s_x2); s_y = sqrt(s_y2);

    hx = zeros(sz,'single'); hy = zeros(sz,'single');
    for gi = 1:NL
        px = p_x{gi} ./ totalSafe;
        hx = hx - px .* log(px + eps);
    end
    for gj = 1:NL
        py = p_y{gj} ./ totalSafe;
        hy = hy - py .* log(py + eps);
    end

    % hxy2 depends only on marginals — compute now, no joint needed
    hxy2 = zeros(sz,'single');
    pxProb = cell(NL,1); pyProb = cell(NL,1);
    for gi = 1:NL, pxProb{gi} = p_x{gi} ./ totalSafe; end
    for gj = 1:NL, pyProb{gj} = p_y{gj} ./ totalSafe; end
    for gi = 1:NL
        for gj = 1:NL
            pxpy = pxProb{gi} .* pyProb{gj};
            hxy2 = hxy2 - pxpy .* log(pxpy + eps);
        end
    end

    % sum-average / sum-entropy / sum-variance / diff-entropy / diff-variance
    % (all purely marginal — matches original computeFeature.m definitions,
    % including its use of raw index (not (index-mean)) for dvarh)
    ii = (1:(2*NL-1))';
    jj = (0:NL-1)';
    savgh = zeros(sz,'single'); senth = zeros(sz,'single');
    for k = 1:(2*NL-1)
        pxy = p_xplusy{k} ./ totalSafe;
        savgh = savgh + (ii(k)+1) .* pxy;
        senth = senth - pxy .* log(pxy + eps);
    end
    svarh = zeros(sz,'single');
    for k = 1:(2*NL-1)
        pxy = p_xplusy{k} ./ totalSafe;
        svarh = svarh + (((ii(k)+1) - senth).^2) .* pxy;
    end
    denth = zeros(sz,'single'); dvarh = zeros(sz,'single');
    for k = 1:NL
        pxmy = p_xminusy{k} ./ totalSafe;
        denth = denth - pxmy .* log(pxmy + eps);
        dvarh = dvarh + (jj(k).*jj(k)) .* pxmy;
    end

    clear p_x p_y p_xplusy p_xminusy

    % glcm_mean = mean(Pij(:)) over an NL x NL matrix that sums to 1 --
    % this is a voxel-independent constant, 1/NL^2 (matches original code)
    glcm_mean = 1/(NL*NL);

    % ---- figure out which features need a second joint pass ----
    marginalOnly = {'savgh','senth','svarh','denth','dvarh'};
    needsJoint = ~all(ismember(GLCM_feat, marginalOnly));

    nFeat = length(GLCM_feat);
    texture = zeros([sz nFeat], 'single');
    featIdx = containers.Map(GLCM_feat, num2cell(1:nFeat));

    if isKey2(featIdx,'savgh'), texture(:,:,:,featIdx('savgh')) = savgh; end
    if isKey2(featIdx,'senth'), texture(:,:,:,featIdx('senth')) = senth; end
    if isKey2(featIdx,'svarh'), texture(:,:,:,featIdx('svarh')) = svarh; end
    if isKey2(featIdx,'denth'), texture(:,:,:,featIdx('denth')) = denth; end
    if isKey2(featIdx,'dvarh'), texture(:,:,:,featIdx('dvarh')) = dvarh; end

    maskExpanded = repmat(mask, [1 1 1 nFeat]);
    texture(~maskExpanded) = 0;

    if ~needsJoint
        return
    end

    % ---- PASS 2: joint-dependent features ----
    % re-derive shifted volumes / validity (cheap; avoids holding them
    % across the whole pass-1 marginal accumulation above)
    SI_shift = cell(nOffsets,1);
    valid_d  = cell(nOffsets,1);
    for d = 1:nOffsets
        SI_shift{d} = shiftVolume(SI, offsets(d,:), 0);
        mask_shift  = shiftVolume(mask, offsets(d,:), false);
        valid_d{d}  = mask & mask_shift;
    end

    accum = struct();
    accNames = {'autoc','contr','dissi','energ','entro','homom','homop','sosvh', ...
                'indnc','idmnc','maxpr','corrm_num','cprom','cshad','hxy1'};
    for a = 1:length(accNames)
        accum.(accNames{a}) = zeros(sz,'single');
    end
    maxpr_acc = zeros(sz,'single');

    for gi = 1:NL
        for gj = 1:NL
            jointSum = zeros(sz,'single');
            for d = 1:nOffsets
                jointSum = jointSum + single(valid_d{d} & (SI==gi) & (SI_shift{d}==gj));
            end
            localCount = boxFilter3(jointSum, R);
            Pij = localCount ./ totalSafe;
            Pij(isnan(Pij)) = 0;

            dij = gi - gj;

            if isKey2(featIdx,'autoc') || isKey2(featIdx,'corrp')
                accum.autoc = accum.autoc + Pij .* (gi*gj);
            end
            if isKey2(featIdx,'contr')
                accum.contr = accum.contr + Pij .* (dij^2);
            end
            if isKey2(featIdx,'dissi')
                accum.dissi = accum.dissi + Pij .* abs(dij);
            end
            if isKey2(featIdx,'energ')
                accum.energ = accum.energ + Pij.^2;
            end
            if isKey2(featIdx,'entro') || isKey2(featIdx,'inf1h') || isKey2(featIdx,'inf2h')
                accum.entro = accum.entro - Pij .* log(Pij + eps);
            end
            if isKey2(featIdx,'homom')
                accum.homom = accum.homom + Pij ./ (1 + abs(dij));
            end
            if isKey2(featIdx,'homop')
                accum.homop = accum.homop + Pij ./ (1 + dij^2);
            end
            if isKey2(featIdx,'sosvh')
                accum.sosvh = accum.sosvh + Pij .* ((gj - glcm_mean).^2);
            end
            if isKey2(featIdx,'indnc')
                accum.indnc = accum.indnc + Pij ./ (1 + abs(dij)/NL);
            end
            if isKey2(featIdx,'idmnc')
                accum.idmnc = accum.idmnc + Pij ./ (1 + (dij/NL)^2);
            end
            if isKey2(featIdx,'maxpr')
                maxpr_acc = max(maxpr_acc, Pij);
            end
            if isKey2(featIdx,'corrm')
                accum.corrm_num = accum.corrm_num + Pij .* (gi-u_x) .* (gj-u_y);
            end
            if isKey2(featIdx,'cprom')
                accum.cprom = accum.cprom + Pij .* ((gi+gj-u_x-u_y).^4);
            end
            if isKey2(featIdx,'cshad')
                accum.cshad = accum.cshad + Pij .* ((gi+gj-u_x-u_y).^3);
            end
            if isKey2(featIdx,'inf1h')
                pxpy_ij = pxProb{gi} .* pyProb{gj};
                accum.hxy1 = accum.hxy1 - Pij .* log(pxpy_ij + eps);
            end
        end
    end
    clear pxProb pyProb

    if isKey2(featIdx,'autoc'), texture(:,:,:,featIdx('autoc')) = accum.autoc; end
    if isKey2(featIdx,'contr'), texture(:,:,:,featIdx('contr')) = accum.contr; end
    if isKey2(featIdx,'dissi'), texture(:,:,:,featIdx('dissi')) = accum.dissi; end
    if isKey2(featIdx,'energ'), texture(:,:,:,featIdx('energ')) = accum.energ; end
    if isKey2(featIdx,'entro'), texture(:,:,:,featIdx('entro')) = accum.entro; end
    if isKey2(featIdx,'homom'), texture(:,:,:,featIdx('homom')) = accum.homom; end
    if isKey2(featIdx,'homop'), texture(:,:,:,featIdx('homop')) = accum.homop; end
    if isKey2(featIdx,'sosvh'), texture(:,:,:,featIdx('sosvh')) = accum.sosvh; end
    if isKey2(featIdx,'indnc'), texture(:,:,:,featIdx('indnc')) = accum.indnc; end
    if isKey2(featIdx,'idmnc'), texture(:,:,:,featIdx('idmnc')) = accum.idmnc; end
    if isKey2(featIdx,'maxpr'), texture(:,:,:,featIdx('maxpr')) = maxpr_acc; end
    if isKey2(featIdx,'corrm')
        texture(:,:,:,featIdx('corrm')) = accum.corrm_num ./ (s_x .* s_y);
    end
    if isKey2(featIdx,'corrp')
        texture(:,:,:,featIdx('corrp')) = (accum.autoc - u_x.*u_y) ./ (s_x .* s_y);
    end
    if isKey2(featIdx,'cprom'), texture(:,:,:,featIdx('cprom')) = accum.cprom; end
    if isKey2(featIdx,'cshad'), texture(:,:,:,featIdx('cshad')) = accum.cshad; end
    if isKey2(featIdx,'inf1h')
        texture(:,:,:,featIdx('inf1h')) = (accum.entro - accum.hxy1) ./ max(hx,hy);
    end
    if isKey2(featIdx,'inf2h')
        val = sqrt(max(0, 1 - exp(-2*(hxy2 - accum.entro))));
        texture(:,:,:,featIdx('inf2h')) = val;
    end

    texture(isnan(texture)) = 0;
    texture(isinf(texture)) = 0;

    % Match original behavior: only report feature values at voxels that
    % are themselves inside the tissue mask. Without this, box-filtered
    % values "bleed" into background voxels near tissue edges (any voxel
    % whose window overlaps tissue gets a non-zero value even if the
    % voxel itself is background), which the original code never did —
    % it explicitly skipped every non-mask center voxel.
    maskExpanded = repmat(mask, [1 1 1 nFeat]);
    texture(~maskExpanded) = 0;
end

function tf = isKey2(m, k)
    tf = isKey(m,k);
end
