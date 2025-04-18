function data = readPCBDefectAnnotations(filename)
    % Read VOC-style XML using xml2struct and output full annotation struct
    x = xml2struct(filename);
    
    if ~isfield(x.annotation, 'object')
        data = struct("filename", "", "Boxes", {[]}, "Labels", {categorical([])});
        return;
    end
    
    % Wrap objects
    if iscell(x.annotation.object)
        objs = x.annotation.object;
    else
        objs = {x.annotation.object};
    end
    
    numObjs = numel(objs);
    boxes = zeros(numObjs, 4);
    labels = categorical(strings(numObjs, 1));
    
    for i = 1:numObjs
        obj = objs{i};
        % Extract label safely
        rawLabel = obj.name.Text;
        labels(i) = categorical(lower(strrep(rawLabel, " ", "_")));
        
        % Extract bounding box
        bbox = obj.bndbox;
        xmin = str2double(bbox.xmin.Text);
        ymin = str2double(bbox.ymin.Text);
        xmax = str2double(bbox.xmax.Text);
        ymax = str2double(bbox.ymax.Text);
        boxes(i, :) = [xmin, ymin, xmax - xmin, ymax - ymin];
    end
    
    % Full image path
    imgFile = x.annotation.filename.Text;
    defectType = x.annotation.folder.Text;
    
    datasetRoot = fileparts(fileparts(fileparts(filename)));  % gets: ...\PCB-DATASET-master
    imgPath = fullfile(datasetRoot, "images", defectType, imgFile);

    
    % Return boxes & labels
    data = struct( ...
        "filename", string(imgPath), ...
        "Boxes", boxes, ...
        "Labels", labels ...
    );
end