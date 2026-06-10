%separate timepoints from stack

file = "G:\FluorescentCollagen\20250518_otherpos\sub7500_flucol594_kpcclusters_800nm_poc0_60745_20x_g55_1_MMStack_Pos4.ome-1.tif";
name = "sub7500_flucol594_kpcclusters_800nm_poc0_60745_20x_g55_1_MMStack_Pos4.ome-1.tif";
outpath = "G:\FluorescentCollagen\20250518_otherpos\maskapplied_timeseries_stacks\";

imgpg1 = imread(file);
[numrow, numcols, ~]  = size(imgpg1);
info = imfinfo(file);
numpages = length(info);
numpages = numpages/25;

for time = 1:25
    
    filenameout = join([outpath name '_t' num2str(time) '.tif']);
    startpage = (time-1)*28 +1;
    for z = 1:28
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
    

    