folder = 'G:\FluorescentCollagen\20260519_flucol_kpc_ows3\20260519_hoescht_col_channels\';
outpath = "G:\FluorescentCollagen\20260519_flucol_kpc_ows3\20260519_hoescht_col_channels\maskapplied_timeseries_stacks\";
numz = 46;
files = dir(fullfile(folder, 'sub7000*.tif'));

for k = 1:numel(files)

    % Full path to input file
    file = fullfile(files(k).folder, files(k).name);

    % Original filename
    name = files(k).name;

    % Filename without extension
    [~, baseName, ~] = fileparts(name);

    fprintf('Processing %s\n', name);


%file = "G:\FluorescentCollagen\20260519_flucol_kpc_ows3\20260519_hoescht_col_channels\sub7000C1-flucol594_bkokpcwhoescht_800nm_blank_44530_37010_blank_g558555_poc0_1_MMStack_Pos2.tif"
%name = "sub7000C1-flucol594_bkokpcwhoescht_800nm_blank_44530_37010_blank_g558555_poc0_1_MMStack_Pos1.tif";

    imgpg1 = imread(file);
    [numrow, numcols, ~]  = size(imgpg1);
    info = imfinfo(file);
    numpages = length(info);
    numpages = numpages/25;
    
    for time = 1:25
        
        filenameout = join([outpath name '_t' num2str(time) '.tif']);
        startpage = (time-1)*numz +1;
        for z = 1:numz
            page = startpage +z -1;
    
            volimg = imread(file,page);
            volimg = double(volimg);
            disp(size(volimg))
            
            if z ==1
                imwrite(uint16(volimg),filenameout);
            else
                imwrite(uint16(volimg), filenameout,'WriteMode','append')
            end
        end
        
    end
end