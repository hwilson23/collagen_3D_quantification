
% ============================================================================
% Windowed 3D GLCM texture pipeline — box-filter version, memory-safe

%with square kernel
%
% Changes vs. the previous box-filter version:
%   1. PRECISION: the integral image / running sums are done in double
%      (exact for integer counts). Counts are stored as single only when
%      exactly representable (< 2^24, asserted). All derived scalars,
%      probabilities and feature accumulators are double.
%   2. MEMORY GUARD: the volume is processed in slabs along its longest
%      axis, with a halo of (R + D) voxels so results are IDENTICAL to
%      whole-volume processing. Slab thickness is chosen from the RAM
%      currently available (/proc/meminfo + cgroup limit). If memory is
%      still short before a slab, it waits, then skips that job with a
%      warning instead of letting Linux run out of RAM.
%   3. FILE LIST: filelist.txt records every finished output. Outputs
%      already listed (or already present in outpath) are not recomputed.
%      Files are written to *.part and renamed only when complete, and the
%      list is updated only after the rename.
% ============================================================================

%% ---------------------- user input ----------------------
HOWMANYZ = 71;
OPT.D                = 1;
OPT.NeighborSizeList = [7];
OPT.quantLevel       = 8;
OPT.glcm_properties  = {'autoc','contr','corrm','corrp','cprom','cshad','dissi', ...
                        'energ','entro','homom','homop','maxpr','sosvh','savgh', ...
                        'svarh','svarc','senth','dvarh','denth','inf1h','inf2h','indnc','idmnc'};

% memory guard settings
OPT.memFraction = 0.60;   % plan to use at most this fraction of AVAILABLE RAM
OPT.minFreeGB   = 2;      % always leave at least this much RAM free
OPT.maxWaitSec  = 120;    % how long to wait for RAM to free up before skipping a job

% disk guard settings
OPT.minFreeDiskGB = 50;     % never let free space on the output drive drop below this
OPT.diskWaitSec   = 1;   % wait this long for you to free space (move files), then stop cleanly
OPT.compress      = true;   % Deflate-compress TIFFs (background zeros shrink a lot)

%folder   = "/home/hwilson23/Documents/MATLAB_processing/maskapplied_timeseries_stacks";
folder = "/home/hwilson23/Documents/MATLAB_processing/maskapplied_timeseries_stacks_231s";
outpath  = "/media/hwilson23/9a2ae3cc-a5be-40d4-ba72-c5f598d7d2b1/helen/output_bksub_texture3d_231newscript";
if ~isfolder(outpath), mkdir(outpath); end
% Keep the list OUTSIDE outpath (or at least don't move it with the tifs):
listFile = "/home/hwilson23/Documents/MATLAB_processing/filelist.txt";

folder = char(folder); outpath = char(outpath); listFile = char(listFile);
cd(folder)

time = 1;
files = dir(fullfile(folder, '*.tif'));
filenames = {files.name};

fprintf('Starting Time: %s\n', datestr(now));

%% ---------------------- file list of finished outputs ----------------------
doneSet = loadFileList(listFile);
isDone  = @(nm) isKey(doneSet, nm) || isfile(fullfile(outpath, nm));
fprintf('%d outputs already listed in %s\n', doneSet.Count, listFile);

% leftover partial files from an earlier crash / full disk are never valid
stale = dir(fullfile(outpath, '*.part'));
for k = 1:numel(stale)
    delete(fullfile(outpath, stale(k).name));
    fprintf('Deleted stale partial file %s\n', stale(k).name);
end

%% ---------------------- global intensity percentile range ----------------------
% NOTE: keep this over ALL input files (even ones already processed) so the
% quantization stays identical between runs.
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

%% ---------------------- main loop ---------------------
fprintf('RAM available: %.1f GB | Disk free on outpath: %.1f GB\n', ...
    getAvailableMemBytes()/2^30, getFreeDiskBytes(outpath)/2^30);
stopRun = false;
for i = 1:length(filenames)

    [~, justname, ~] = fileparts(filenames{i});

    % ---- skip the whole input if every output already exists ----
    pending = false;
    for R = OPT.NeighborSizeList
        for prop = 1:numel(OPT.glcm_properties)
            nm = makeOutName(justname, OPT.glcm_properties{prop}, OPT.D, R, OPT.quantLevel);
            if ~isDone(nm), pending = true; end
        end
    end
    if ~pending
        fprintf('Skipping %s (all outputs already done)\n', justname);
        continue
    end

    disp(filenames{i})
    imgpg1 = imread(fullfile(folder, filenames{i}));
    [numRows, numCols, ~] = size(imgpg1);

    volimg = zeros(numRows, numCols, HOWMANYZ, class(imgpg1));
    startPage = (time - 1) * HOWMANYZ + 1;
    for z = 1:HOWMANYZ
        volimg(:,:,z) = imread(fullfile(folder, filenames{i}), startPage + z - 1);
    end
    clear imgpg1
    volimg = double(volimg);
    disp(size(volimg))
    origsize = size(volimg);

    mask = volimg > 0;

    % bounding box of tissue (projections instead of ind2sub: no huge index vectors)
    rr = find(any(any(mask,2),3));
    cc = find(any(any(mask,1),3));
    zz = find(any(any(mask,1),2));
    if isempty(rr)
        fprintf('No foreground in %s, skipping\n', justname);
        continue
    end
    rRange = rr(1):rr(end); cRange = cc(1):cc(end); zRange = zz(1):zz(end);

    cropMask = mask(rRange, cRange, zRange);
    cropVol  = volimg(rRange, cRange, zRange);
    clear volimg mask

    % quantize ONCE in double precision, then drop the double volume
    SI = quantizeVolume(cropVol, OPT.quantLevel, [setmin, setmax]);
    clear cropVol

    for R = OPT.NeighborSizeList

        names = cell(1, numel(OPT.glcm_properties));
        for prop = 1:numel(names)
            names{prop} = makeOutName(justname, OPT.glcm_properties{prop}, OPT.D, R, OPT.quantLevel);
        end
        todo = find(~cellfun(isDone, names));
        if isempty(todo)
            fprintf('Skipping %s | R=%d (done)\n', justname, R);
            continue
        end
        feats = OPT.glcm_properties(todo);   % only compute what is missing

        % make sure the outputs will fit BEFORE spending time computing them
        % (uncompressed size is used as a conservative upper bound)
        if ~ensureDisk(outpath, numel(todo) * prod(origsize) * 4, OPT.minFreeDiskGB * 2^30, OPT.diskWaitSec)
            stopRun = true; break
        end

        timerStart = tic;
        fprintf('Processing %s | NL=%d | R=%d | %d feature(s)\n', justname, OPT.quantLevel, R, numel(feats));

        try
            texture = computeChunked(SI, cropMask, OPT.quantLevel, OPT.D, R, feats, OPT);
        catch ME
            if strcmp(ME.identifier, 'GLCM:memory')
                warning('Skipping %s R=%d: %s', justname, R, ME.message);
                continue
            else
                rethrow(ME);
            end
        end
        fprintf('Total time (NL=%d, R=%d): %2.2f s\n', OPT.quantLevel, R, toc(timerStart));

        for k = 1:numel(todo)
            fullimg = zeros(origsize, 'single');
            fullimg(rRange, cRange, zRange) = texture(:,:,:,k);

            finalPath = fullfile(outpath, names{todo(k)});
            tmpPath   = [finalPath '.part'];
            
            if ~ensureDisk(outpath, numel(fullimg) * 4, OPT.minFreeDiskGB * 2^30, OPT.diskWaitSec)
                stopRun = true; break
            end
            try
                writeTiffStack(tmpPath, fullimg, OPT.compress);
                movefile(tmpPath, finalPath);          % only complete files get the real name
            catch ME
                if isfile(tmpPath), delete(tmpPath); end
                rethrow(ME);
            end
            appendToFileList(listFile, names{todo(k)}); % update list AFTER successful write
            doneSet(names{todo(k)}) = true;
            clear fullimg
        end
        if stopRun, break; end
        clear texture
    end
    if stopRun, break;end
    clear SI cropMask
    disp("file saved!")
end

if stopRun, disp("stopped early: low disk space"); else, disp("all done"); end
fprintf('End Time: %s\n', datestr(now));


% ============================================================================
%                              FILE LIST HELPERS
% ============================================================================

function nm = makeOutName(justname, prop, D, R, Q)
nm = sprintf('%s_%s_3D_D%d_R%d_Q%d_raw.tif', justname, prop, D, R, Q);
end

function s = loadFileList(listFile)
    s = containers.Map('KeyType','char','ValueType','logical');
    if ~isfile(listFile), return; end
    fid = fopen(listFile, 'r');
    c = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', '');
    fclose(fid);
    lines = strtrim(c{1});
    for k = 1:numel(lines)
        if ~isempty(lines{k}), s(lines{k}) = true; end
    end
end

function appendToFileList(listFile, nm)
    fid = fopen(listFile, 'a');
    if fid < 0, error('Cannot open file list %s for appending', listFile); end
    fprintf(fid, '%s\n', nm);
    fclose(fid);
end

function writeTiffStack(path, vol, compress)
    if nargin < 3, compress = false; end
    t = Tiff(path, 'w');
    cleaner = onCleanup(@() close(t));   %#ok<NASGU>  closes even if a write fails
    for a = 1:size(vol,3)
        slice = vol(:,:,a);
        tag = struct();
        tag.ImageLength = size(slice,1);
        tag.ImageWidth  = size(slice,2);
        tag.Photometric = Tiff.Photometric.MinIsBlack;
        tag.BitsPerSample = 32;
        tag.SamplesPerPixel = 1;
        tag.SampleFormat = Tiff.SampleFormat.IEEEFP;
        tag.RowsPerStrip = size(slice,1);
        tag.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
        if compress
            tag.Compression = Tiff.Compression.AdobeDeflate;   % lossless
        end
        t.setTag(tag);
        t.write(slice);
        if a < size(vol,3), t.writeDirectory(); end
    end
end


% ============================================================================
%                              MEMORY HELPERS
% ============================================================================
function b = getAvailableMemBytes()
% RAM that can be used without swapping, in bytes. Reads MemAvailable from
% /proc/meminfo and also respects a cgroup (container/systemd) limit.
    persistent warned
    b = Inf;
    try
        fid = fopen('/proc/meminfo','r');
        while true
            ln = fgetl(fid);
            if ~ischar(ln), break; end
            tok = regexp(ln, '^MemAvailable:\s+(\d+)\s+kB', 'tokens', 'once');
            if ~isempty(tok), b = str2double(tok{1}) * 1024; break; end
        end
        fclose(fid);
    catch
    end
    try   % cgroup v2
        lim = strtrim(fileread('/sys/fs/cgroup/memory.max'));
        if ~strcmp(lim, 'max')
            cur = str2double(strtrim(fileread('/sys/fs/cgroup/memory.current')));
            b = min(b, str2double(lim) - cur);
        end
    catch
    end 
    if isinf(b) && isempty(warned)
        warning('Could not determine available memory; memory guard is disabled.');
        warned = true;
    end
end

function ensureMemory(neededBytes, minFreeBytes, maxWaitSec)
    t0 = tic;
    while true
        avail = getAvailableMemBytes();
        
        if avail >= neededBytes + minFreeBytes, return; end
        if toc(t0) > maxWaitSec
            error('GLCM:memory', ['need ~%.1f GB (+%.1f GB reserve) but only %.1f GB is ' ...
                  'available. Close other programs, lower quantLevel/R, or request fewer features.'], ...
                  neededBytes/2^30, minFreeBytes/2^30, avail/2^30);
        end
        fprintf('  waiting for RAM (need %.1f GB, have %.1f GB)...\n', ...
                (neededBytes+minFreeBytes)/2^30, avail/2^30);
        pause(10);
    end
end

function b = getFreeDiskBytes(p)
    b = NaN;
    try
        [st, out] = system(sprintf('df -Pk "%s"', p));
        if st == 0
            lines = strsplit(strtrim(out), newline);
            f = strsplit(strtrim(lines{end}));
            b = str2double(f{4}) * 1024;
        end
    catch
    end
    if isnan(b)
        error('GLCM:disk', 'Could not read free disk space for %s; stopping for safety.', p);
    end
end

function ok = ensureDisk(p, neededBytes, reserveBytes, maxWaitSec)
    t0 = tic; ok = true;
    while true
        free = getFreeDiskBytes(p);
        if free >= neededBytes + reserveBytes, return; end
        if toc(t0) >= maxWaitSec
            fprintf(['STOPPING: %.1f GB free on output drive, need %.1f GB + %.1f GB reserve.\n' ...
                     'Move finished files elsewhere and re-run; completed outputs are in the file list.\n'], ...
                     free/2^30, neededBytes/2^30, reserveBytes/2^30);
            ok = false; return;
        end
        fprintf('  disk low: %.1f GB free. Retrying in 60 s...\n', free/2^30);
        pause(60);
    end
end

function b = estimateBytesPerVoxel(NL, nOff, nFeat)
% Rough PEAK bytes per voxel of one slab inside computeWindowedGLCMFeatures.
    b = 2;                  % SI (uint8), mask
    b = b + 2*nOff;         % SI_shift (uint8) + valid_d (logical) per offset
    b = b + 28*NL;          % marginals: single counts + double probabilities
    b = b + 8*22;           % double scalars: totalSafe,u,s,hx,hy,hxy2,savgh,... 
    b = b + 4*nFeat;        % slab texture (single)
    b = b + 8*17;           % feature accumulators (double)
    b = b + 100;            % transients: box filter copies, jointSum, Pij, temporaries
end


% ============================================================================
%                              CORE FUNCTIONS
% ============================================================================
function offset = ComputeOffsets(dis)
    [delta1,delta2,delta3] = ndgrid(-dis:dis,-dis:dis,-dis:dis);
    inside = round(sqrt(delta1.*delta1+delta2.*delta2+delta3.*delta3))<=dis;
    offset = [delta1(inside(:)),delta2(inside(:)),delta3(inside(:))];
    offset(offset(:,1)==0&offset(:,2)==0&offset(:,3)==0,:) = [];
end

function SI = quantizeVolume(vol, NL, GL)
% Quantize in double precision (same math as the original CreateGLCM_Local).
    if GL(2) == GL(1)
        SI = ones(size(vol));
    else
        slope = (NL-1) / (GL(2)-GL(1));
        intercept = 1 - slope*GL(1);
        SI = round(vol.*slope + intercept);
    end
    SI(SI > NL) = NL;
    SI(SI < 1)  = 1;
    if NL <= 255, SI = uint8(SI); else, SI = uint16(SI); end
end

function out = boxFilter3(in, R)
% Sum of `in` over a cubic window (radius R, zero padded), computed as three
% separable running-sum passes in DOUBLE precision (exact for integer data).
% Returns single: callers guarantee counts < 2^24 so single is exact.
    out = double(in);
    for dim = 1:3
        out = boxFilter1(out, R, dim);
    end
    out = single(out);
end

function out = boxFilter1(in, R, dim)
    n   = size(in, dim);
    win = 2*R + 1;
    padv = zeros(1,3); padv(dim) = R;
    prev = zeros(1,3); prev(dim) = 1;
    C = cumsum(padarray(in, padv, 0, 'both'), dim);
    C = padarray(C, prev, 0, 'pre');
    idxHi = {':',':',':'}; idxHi{dim} = (1:n) + win;
    idxLo = {':',':',':'}; idxLo{dim} = 1:n;
    out = C(idxHi{:}) - C(idxLo{:});
end

function out = shiftVolume(vol, offset, fillVal)
% vol sampled at pos+offset, fillVal outside the volume.
    sz = size(vol);
    if islogical(vol)
        out = false(sz);
    else
        out = cast(zeros(sz), 'like', vol);
        if fillVal ~= 0, out(:) = fillVal; end
    end
    r = 1:sz(1); c = 1:sz(2); p = 1:sz(3);
    rs = r + offset(1); cs = c + offset(2); ps = p + offset(3);
    validR = rs>=1 & rs<=sz(1);
    validC = cs>=1 & cs<=sz(2);
    validP = ps>=1 & ps<=sz(3);
    out(validR,validC,validP) = vol(rs(validR), cs(validC), ps(validP));
end

function texture = computeChunked(SI, mask, NL, D, R, feats, OPT)
% Memory-aware driver: processes slabs along the longest axis with a halo of
% R + D voxels, so every voxel sees its full neighborhood (no seams).
    sz    = size(SI);
    nFeat = numel(feats);
    nOff  = size(ComputeOffsets(D), 1);
    halo  = R + D;
    bpv   = estimateBytesPerVoxel(NL, nOff, nFeat) * 1.25;   % 25% safety margin
    minFreeBytes = OPT.minFreeGB * 2^30;

    avail  = getAvailableMemBytes();
    budget = min(OPT.memFraction * avail, avail - minFreeBytes);

    texBytes = 4 * nFeat * numel(SI);
    if texBytes > budget
        error('GLCM:memory', ['result array alone needs %.1f GB but the budget is %.1f GB; ' ...
              'request fewer features or crop the volume'], texBytes/2^30, budget/2^30);
    end
    texture = zeros([sz nFeat], 'single');
    budget  = budget - texBytes;

    [~, cdim] = max(sz);
    n = sz(cdim);
    planeVox = numel(SI) / n;
    maxSlab  = floor(budget / (bpv * planeVox));    % slab thickness incl. halos
    core     = min(maxSlab - 2*halo, n);
    if core < 1
        error('GLCM:memory', ['even a minimal slab (%d planes) needs ~%.1f GB but the budget is ' ...
              '%.1f GB; lower quantLevel or R'], 2*halo+1, bpv*planeVox*(2*halo+1)/2^30, budget/2^30);
    end
    fprintf('  memory: avail %.1f GB, budget %.1f GB -> slabs of %d planes (+/-%d halo) along dim %d\n', ...
            avail/2^30, budget/2^30, core, halo, cdim);

    for s = 1:core:n
        e  = min(s + core - 1, n);
        lo = max(1, s - halo);
        hi = min(n, e + halo);

        idxCore = {':',':',':'}; idxCore{cdim} = s:e;
        if ~any(mask(idxCore{:}), 'all'), continue; end   % nothing to compute

        idxSlab = {':',':',':'}; idxSlab{cdim} = lo:hi;
        subSI   = SI(idxSlab{:});
        subMask = mask(idxSlab{:});

        ensureMemory(bpv * numel(subSI), minFreeBytes, OPT.maxWaitSec);

        subTex = computeWindowedGLCMFeatures(subSI, subMask, NL, D, R, feats);

        idxSrc = {':',':',':',':'}; idxSrc{cdim} = (s-lo+1):(e-lo+1);
        idxDst = {':',':',':',':'}; idxDst{cdim} = s:e;
        texture(idxDst{:}) = subTex(idxSrc{:});
        clear subSI subMask subTex
    end
end

function [SI_shift, valid_d] = buildShifted(SI, mask, offsets)
    nOffsets = size(offsets,1);
    SI_shift = cell(nOffsets,1);
    valid_d  = cell(nOffsets,1);
    for d = 1:nOffsets
        SI_shift{d} = shiftVolume(SI, offsets(d,:), 0);
        valid_d{d}  = mask & shiftVolume(mask, offsets(d,:), false);
    end
end

function texture = computeWindowedGLCMFeatures(SI, mask, NL, D, R, GLCM_feat)
% Two-pass windowed-GLCM features on an already-quantized volume.
% Counts are exact integers (single); all probabilities/features are double.

    sz   = size(SI);
    mask = logical(mask);
    offsets  = ComputeOffsets(D);
    nOffsets = size(offsets,1);

    win = 2*R + 1;
    assert(win^3 * nOffsets < 2^24, 'window count exceeds exact single-precision range');

    [SI_shift, valid_d] = buildShifted(SI, mask, offsets);

    % ---- total valid pair count per voxel ----
    validSum = zeros(sz,'single');
    for d = 1:nOffsets
        validSum = validSum + single(valid_d{d});
    end
    totalSafe = double(boxFilter3(validSum, R));
    totalSafe(totalSafe == 0) = NaN;
    clear validSum

    % ---- PASS 1: marginals (single counts, exact) ----
    p_x = cell(NL,1);  p_y = cell(NL,1);
    for k = 1:NL, p_x{k} = zeros(sz,'single'); p_y{k} = zeros(sz,'single'); end
    p_xplusy  = cell(2*NL-1,1);
    for k = 1:(2*NL-1), p_xplusy{k} = zeros(sz,'single'); end
    p_xminusy = cell(NL,1);
    for k = 1:NL, p_xminusy{k} = zeros(sz,'single'); end

    for gi = 1:NL
        Mi = (SI == gi);
        for gj = 1:NL
            jointSum = zeros(sz,'single');
            for d = 1:nOffsets
                jointSum = jointSum + single(valid_d{d} & Mi & (SI_shift{d} == gj));
            end
            if ~any(jointSum(:)), continue; end       % contributes exactly zero
            localCount = boxFilter3(jointSum, R);

            p_x{gi} = p_x{gi} + localCount;
            p_y{gj} = p_y{gj} + localCount;
            k1 = gi + gj - 1;
            k2 = abs(gi - gj) + 1;
            p_xplusy{k1}  = p_xplusy{k1}  + localCount;
            p_xminusy{k2} = p_xminusy{k2} + localCount;
        end
    end
    clear jointSum localCount valid_d SI_shift Mi

    % ---- convert row/col marginals to double probabilities (in place) ----
    for k = 1:NL
        p_x{k} = double(p_x{k}) ./ totalSafe;
        p_y{k} = double(p_y{k}) ./ totalSafe;
    end

    u_x = zeros(sz); u_y = zeros(sz);
    for k = 1:NL
        u_x = u_x + k .* p_x{k};
        u_y = u_y + k .* p_y{k};
    end
    s_x2 = zeros(sz); s_y2 = zeros(sz);
    for k = 1:NL
        s_x2 = s_x2 + ((k-u_x).^2) .* p_x{k};
        s_y2 = s_y2 + ((k-u_y).^2) .* p_y{k};
    end
    s_x = sqrt(s_x2); s_y = sqrt(s_y2);
    clear s_x2 s_y2

    hx = zeros(sz); hy = zeros(sz);
    for k = 1:NL
        hx = hx - p_x{k} .* log(p_x{k} + eps);
        hy = hy - p_y{k} .* log(p_y{k} + eps);
    end

    hxy2 = zeros(sz);
    for gi = 1:NL
        for gj = 1:NL
            pxpy = p_x{gi} .* p_y{gj};
            hxy2 = hxy2 - pxpy .* log(pxpy + eps);
        end
    end

    % sum / difference statistics (definitions match the original computeFeature)
    savgh = zeros(sz); senth = zeros(sz);
    for k = 1:(2*NL-1)
        pxy = double(p_xplusy{k}) ./ totalSafe;
        savgh = savgh + (k+1) .* pxy;
        senth = senth - pxy .* log(pxy + eps);
    end
    svarh = zeros(sz);
    for k = 1:(2*NL-1)
        pxy = double(p_xplusy{k}) ./ totalSafe;
        svarh = svarh + (((k+1) - senth).^2) .* pxy;
    end
    svarc = zeros(sz);
    for k = 1:(2*NL-1)
        pxy = double(p_xplusy{k}) ./ totalSafe;
        svarc = svarc + (((k+1) - savgh).^2) .* pxy;
    end
    denth = zeros(sz); dvarh = zeros(sz);
    for k = 1:NL
        pxmy = double(p_xminusy{k}) ./ totalSafe;
        denth = denth - pxmy .* log(pxmy + eps);
        dvarh = dvarh + ((k-1)^2) .* pxmy;
    end
    clear p_xplusy p_xminusy pxy pxmy pxpy

    glcm_mean = 1/(NL*NL);

    % ---- assign marginal-only features ----
    nFeat = numel(GLCM_feat);
    texture = zeros([sz nFeat], 'single');
    featIdx = containers.Map(GLCM_feat, num2cell(1:nFeat));
    has = @(f) isKey(featIdx, f);

    if has('savgh'), texture(:,:,:,featIdx('savgh')) = savgh; end
    if has('senth'), texture(:,:,:,featIdx('senth')) = senth; end
    if has('svarh'), texture(:,:,:,featIdx('svarh')) = svarh; end
    if has('svarc'), texture(:,:,:,featIdx('svarc')) = svarc; end
    if has('denth'), texture(:,:,:,featIdx('denth')) = denth; end
    if has('dvarh'), texture(:,:,:,featIdx('dvarh')) = dvarh; end
    clear savgh senth svarh svarc denth dvarh

    marginalOnly = {'savgh','senth','svarh','svarc','denth','dvarh'};
    if all(ismember(GLCM_feat, marginalOnly))
        texture = finalizeTexture(texture, mask);
        return
    end

    % ---- which accumulators are actually needed (allocate only those) ----
    need.autoc = has('autoc') || has('corrp');
    need.contr = has('contr');
    need.dissi = has('dissi');
    need.energ = has('energ');
    need.entro = has('entro') || has('inf1h') || has('inf2h');
    need.homom = has('homom');
    need.homop = has('homop');
    need.sosvh = has('sosvh');
    need.indnc = has('indnc');
    need.idmnc = has('idmnc');
    need.maxpr = has('maxpr');
    need.corrm = has('corrm');
    need.cprom = has('cprom');
    need.cshad = has('cshad');
    need.hxy1  = has('inf1h');
    accNames = fieldnames(need);
    for a = 1:numel(accNames)
        if need.(accNames{a}), acc.(accNames{a}) = zeros(sz); end
    end

    % ---- PASS 2: joint-dependent features ----
    [SI_shift, valid_d] = buildShifted(SI, mask, offsets);

    for gi = 1:NL
        Mi = (SI == gi);
        for gj = 1:NL
            jointSum = zeros(sz,'single');
            for d = 1:nOffsets
                jointSum = jointSum + single(valid_d{d} & Mi & (SI_shift{d} == gj));
            end
            if ~any(jointSum(:)), continue; end       % all terms are multiples of Pij
            Pij = double(boxFilter3(jointSum, R)) ./ totalSafe;
            Pij(isnan(Pij)) = 0;
            dij = gi - gj;

            if need.autoc, acc.autoc = acc.autoc + Pij .* (gi*gj); end
            if need.contr, acc.contr = acc.contr + Pij .* (dij^2); end
            if need.dissi, acc.dissi = acc.dissi + Pij .* abs(dij); end
            if need.energ, acc.energ = acc.energ + Pij.^2; end
            if need.entro, acc.entro = acc.entro - Pij .* log(Pij + eps); end
            if need.homom, acc.homom = acc.homom + Pij ./ (1 + abs(dij)); end
            if need.homop, acc.homop = acc.homop + Pij ./ (1 + dij^2); end
            if need.sosvh, acc.sosvh = acc.sosvh + Pij .* ((gj - glcm_mean)^2); end
            if need.indnc, acc.indnc = acc.indnc + Pij ./ (1 + abs(dij)/NL); end
            if need.idmnc, acc.idmnc = acc.idmnc + Pij ./ (1 + (dij/NL)^2); end
            if need.maxpr, acc.maxpr = max(acc.maxpr, Pij); end
            if need.corrm, acc.corrm = acc.corrm + Pij .* (gi-u_x) .* (gj-u_y); end
            if need.cprom, acc.cprom = acc.cprom + Pij .* ((gi+gj-u_x-u_y).^4); end
            if need.cshad, acc.cshad = acc.cshad + Pij .* ((gi+gj-u_x-u_y).^3); end
            if need.hxy1,  acc.hxy1  = acc.hxy1  - Pij .* log(p_x{gi} .* p_y{gj} + eps); end
        end
    end
    clear Pij jointSum valid_d SI_shift Mi p_x p_y totalSafe

    simple = {'autoc','contr','dissi','energ','entro','homom','homop','sosvh','indnc','idmnc', ...
              'maxpr','cprom','cshad'};
    for a = 1:numel(simple)
        f = simple{a};
        if has(f), texture(:,:,:,featIdx(f)) = acc.(f); end
    end
    if has('corrm'), texture(:,:,:,featIdx('corrm')) = acc.corrm ./ (s_x .* s_y); end
    if has('corrp'), texture(:,:,:,featIdx('corrp')) = (acc.autoc - u_x.*u_y) ./ (s_x .* s_y); end
    if has('inf1h')
        texture(:,:,:,featIdx('inf1h')) = (acc.entro - acc.hxy1) ./ max(hx, hy);
    end
    if has('inf2h')
        texture(:,:,:,featIdx('inf2h')) = sqrt(max(0, 1 - exp(-2*(hxy2 - acc.entro))));
    end

    texture = finalizeTexture(texture, mask);
end

function t = finalizeTexture(t, mask)
% NaN/Inf -> 0, and zero every voxel outside the tissue mask (one feature at
% a time, so no nFeat-times-larger temporary mask is created).
    for f = 1:size(t,4)
        s = t(:,:,:,f);
        s(~isfinite(s)) = 0;
        s(~mask) = 0;
        t(:,:,:,f) = s;
    end
end